"""分层记忆存储（纯 SQLite + FTS5 trigram）。

为什么不上向量库：单陪伴机器人、单用户，量级小；sqlite 3.50 的 FTS5 + trigram
已实测能对中文做子串检索（「巧克力」可命中「我喜欢酒心巧克力」），
足够用且零新增依赖（方案明确「不引向量库」）。第三期如需可再叠加 embedding。

六张表 + 一张 FTS 虚表（阶段三记忆升级：结构化事实 + 核心记忆）：
    messages     原始对话流水
    sessions     短期记忆：会话摘要
    long_memory  长期记忆（source = chat_summary / profile_fact / self_reflection / manual）
    profile      用户画像（facts 存 JSON）
    entity_facts 结构化事实（Mem0 风格：subject/predicate/object，同槽位覆盖合并）
    core_facts   核心记忆（MemGPT core 风格：始终注入提示词的少量高价值事实）
    memory_fts   fts5(content, keywords, tokenize='trigram')

并发模型：单个 sqlite3 连接（check_same_thread=False）+ threading.Lock，
所有阻塞操作经 asyncio.to_thread 丢到线程里执行，绝不阻塞事件循环
（这是对老项目「async 里直接同步 IO」硬伤的修复）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)

# 未指定用户时使用的画像键
DEFAULT_USER = "default"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    role      TEXT NOT NULL,
    content   TEXT NOT NULL,
    user      TEXT DEFAULT '',
    user_name TEXT DEFAULT '',
    channel   TEXT DEFAULT '',
    ts        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(ts);

CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user       TEXT DEFAULT '',
    summary    TEXT NOT NULL,
    started_ts REAL NOT NULL,
    ended_ts   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_ended ON sessions(ended_ts);

CREATE TABLE IF NOT EXISTS long_memory (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content    TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'manual',
    importance REAL NOT NULL DEFAULT 0.5,
    user       TEXT DEFAULT '',
    created_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_long_created ON long_memory(created_ts);

CREATE TABLE IF NOT EXISTS profile (
    user       TEXT PRIMARY KEY,
    nick       TEXT DEFAULT '',
    facts      TEXT DEFAULT '[]',
    updated_ts REAL NOT NULL
);

-- 结构化事实（Mem0 风格）：把对话抽成 (subject, predicate, object) 三元组式事实，
-- 同一 (subject, predicate) 视为「同一槽位」→ 新值覆盖旧值并累加 updated_count，
-- 从而做到去重 + 冲突消解，而不是把自由文本一条条堆起来。
CREATE TABLE IF NOT EXISTS entity_facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user          TEXT DEFAULT '',
    subject       TEXT NOT NULL,
    predicate     TEXT NOT NULL,
    object        TEXT NOT NULL,
    fact          TEXT NOT NULL DEFAULT '',
    importance    REAL NOT NULL DEFAULT 0.5,
    first_seen    REAL NOT NULL,
    last_seen     REAL NOT NULL,
    updated_count INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_fact_user ON entity_facts(user);
CREATE INDEX IF NOT EXISTS idx_fact_seen ON entity_facts(last_seen);

-- 核心记忆（MemGPT/Letta core memory 风格）：少量「始终注入决策提示词」的高价值事实，
-- 由 Agent 通过 memory_ops 主动 pin 进来；与需要检索的长期记忆分层区分。
CREATE TABLE IF NOT EXISTS core_facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user       TEXT DEFAULT '',
    content    TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.8,
    created_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_core_user ON core_facts(user);

-- FTS5 虚表：trigram 分词支持中文子串；standalone，rowid 与 long_memory.id 对齐
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts
    USING fts5(content, keywords, tokenize='trigram');
"""


class MemoryStore:
    """记忆库。"""

    def __init__(self, db_path: str | Path) -> None:
        self.path = Path(db_path)
        if not self.path.is_absolute():
            # 相对路径统一挂到项目根下
            from ..kernel.config import package_root
            self.path = package_root() / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL：读写并发更友好
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("记忆库就绪：%s", self.path)

    # ------------------------------------------------------------------ 线程封装
    async def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """把阻塞的 sqlite 操作放到线程池执行。"""
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def close(self) -> None:
        await self._call(self._close_sync)

    def _close_sync(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ messages
    async def add_message(self, role: str, content: str, *, user: str = "",
                          user_name: str = "", channel: str = "",
                          ts: float | None = None) -> None:
        """追加一条原始对话流水。"""
        await self._call(self._add_message_sync, role, content, user, user_name,
                         channel, ts if ts is not None else time.time())

    def _add_message_sync(self, role: str, content: str, user: str, user_name: str,
                          channel: str, ts: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages(role, content, user, user_name, channel, ts)"
                " VALUES(?,?,?,?,?,?)",
                (role, content, user, user_name, channel, ts))
            self._conn.commit()

    async def recent_messages(self, limit: int = 20, user: str = "") -> list[dict[str, Any]]:
        """取最近若干条流水（备份视角，工作记忆仍以 SessionContext 为准）。"""
        return await self._call(self._recent_messages_sync, limit, user)

    def _recent_messages_sync(self, limit: int, user: str) -> list[dict[str, Any]]:
        with self._lock:
            if user:
                rows = self._conn.execute(
                    "SELECT * FROM messages WHERE user=? ORDER BY id DESC LIMIT ?",
                    (user, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in reversed(rows)]

    # ------------------------------------------------------------------ sessions
    async def add_session(self, summary: str, *, user: str = "",
                          started_ts: float = 0.0, ended_ts: float | None = None) -> None:
        """写入一条会话摘要。"""
        await self._call(self._add_session_sync, summary, user, started_ts,
                         ended_ts if ended_ts is not None else time.time())

    def _add_session_sync(self, summary: str, user: str, started_ts: float,
                          ended_ts: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions(user, summary, started_ts, ended_ts) VALUES(?,?,?,?)",
                (user, summary, started_ts, ended_ts))
            self._conn.commit()

    async def recent_sessions(self, limit: int = 2, user: str = "") -> list[str]:
        """取最近若干条会话摘要（新的排后面）。"""
        return await self._call(self._recent_sessions_sync, limit, user)

    def _recent_sessions_sync(self, limit: int, user: str) -> list[str]:
        with self._lock:
            if user:
                rows = self._conn.execute(
                    "SELECT summary FROM sessions WHERE user=? ORDER BY ended_ts DESC LIMIT ?",
                    (user, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT summary FROM sessions ORDER BY ended_ts DESC LIMIT ?",
                    (limit,)).fetchall()
        return [r["summary"] for r in reversed(rows)]

    # ------------------------------------------------------------------ long_memory
    async def add_long(self, content: str, *, source: str = "manual",
                       importance: float = 0.5, user: str = "",
                       keywords: str = "") -> int:
        """写入一条长期记忆，并同步进 FTS 索引。返回其 id。"""
        return await self._call(self._add_long_sync, content, source,
                                float(importance), user, keywords)

    def _add_long_sync(self, content: str, source: str, importance: float,
                       user: str, keywords: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO long_memory(content, source, importance, user, created_ts)"
                " VALUES(?,?,?,?,?)",
                (content, source, importance, user, time.time()))
            rowid = cur.lastrowid
            # FTS 用与 long_memory 相同的 rowid，便于 join 回来
            self._conn.execute(
                "INSERT INTO memory_fts(rowid, content, keywords) VALUES(?,?,?)",
                (rowid, content, keywords or content))
            self._conn.commit()
        return int(rowid)

    async def search_long(self, query: str, *, top_k: int = 5,
                          half_life_days: float = 30.0,
                          user: str = "") -> list[str]:
        """检索长期记忆：三因子排序 relevance×importance×recency（见 _search_long_sync）。"""
        return await self._call(self._search_long_sync, query, top_k,
                                half_life_days, user)

    def _search_long_sync(self, query: str, top_k: int, half_life_days: float,
                          user: str) -> list[str]:
        """三因子排序（Generative Agents 风格，Stanford 2023）：

            score = relevance × importance × recency_decay

        之前只用 importance×衰减，相关性只是「命中/不命中」的二值；接入 FTS5 的
        bm25（rank）后，越贴题的旧事越靠前，召回质量明显提升。
        """
        candidates = self._fts_candidates(query)
        if not candidates:
            return []
        now = time.time()
        half_life = max(half_life_days, 1.0) * 86400.0
        scored: list[tuple[float, str]] = []
        with self._lock:
            for rowid, relevance in candidates[:20]:
                row = self._conn.execute(
                    "SELECT content, importance, created_ts, user FROM long_memory WHERE id=?",
                    (rowid,)).fetchone()
                if row is None:
                    continue
                if user and row["user"] not in ("", user):
                    continue
                age = max(now - row["created_ts"], 0.0)
                decay = math.exp(-age / half_life)
                # 三因子相乘：相关度 × 重要度 × 时间衰减
                scored.append((relevance * row["importance"] * decay, row["content"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    def _fts_candidates(self, query: str) -> list[tuple[int, float]]:
        """用 FTS5 找候选 (rowid, 相关度)；无法分词（如单字）时退回 LIKE 扫描。

        相关度来源：FTS5 的 rank 列（bm25，越负越相关）取负后，按候选内最大值
        归一化到 (0,1]，好与 importance、衰减同尺度相乘。LIKE 兜底没有 bm25，
        视为强匹配，相关度给 1.0。
        """
        match_expr = _make_match_expr(query)
        with self._lock:
            if match_expr:
                try:
                    rows = self._conn.execute(
                        "SELECT rowid, rank FROM memory_fts WHERE memory_fts MATCH ?"
                        " ORDER BY rank LIMIT 50",
                        (match_expr,)).fetchall()
                    return _normalize_relevance(
                        [(int(r["rowid"]), -float(r["rank"])) for r in rows])
                except sqlite3.OperationalError as e:
                    logger.warning("FTS 查询失败，退回 LIKE：%s", e)
            # LIKE 兜底：处理 <3 字（trigram 无法匹配）等情况
            like = f"%{query.strip()}%"
            if not query.strip():
                return []
            rows = self._conn.execute(
                "SELECT id FROM long_memory WHERE content LIKE ? ORDER BY created_ts DESC"
                " LIMIT 50", (like,)).fetchall()
        return [(int(r["id"]), 1.0) for r in rows]

    # ------------------------------------------------------------------ 管理端 CRUD
    # 以下方法供 WebUI「记忆管理」页使用（增删改查 + 分页），与上面的写入路径互不影响。
    async def list_long(self, limit: int = 100, offset: int = 0,
                        user: str = "") -> list[dict[str, Any]]:
        """分页列出长期记忆（按创建时间倒序），供后台表格展示。"""
        return await self._call(self._list_long_sync, limit, offset, user)

    def _list_long_sync(self, limit: int, offset: int, user: str) -> list[dict[str, Any]]:
        with self._lock:
            if user:
                rows = self._conn.execute(
                    "SELECT * FROM long_memory WHERE user=? ORDER BY id DESC LIMIT ? OFFSET ?",
                    (user, limit, offset)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM long_memory ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset)).fetchall()
        return [dict(r) for r in rows]

    async def update_long(self, mem_id: int, *, content: str | None = None,
                          importance: float | None = None) -> bool:
        """修改一条长期记忆；content 变化时同步刷新 FTS 索引。"""
        return await self._call(self._update_long_sync, mem_id, content, importance)

    def _update_long_sync(self, mem_id: int, content: str | None,
                          importance: float | None) -> bool:
        fields: list[str] = []
        values: list[Any] = []
        if content is not None:
            fields.append("content=?"); values.append(content)
        if importance is not None:
            fields.append("importance=?"); values.append(float(importance))
        if not fields:
            return False
        values.append(mem_id)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE long_memory SET {', '.join(fields)} WHERE id=?", values)
            if content is not None:
                # FTS 是独立虚表，必须手动同步（否则改完搜不到新内容）
                self._conn.execute(
                    "UPDATE memory_fts SET content=?, keywords=? WHERE rowid=?",
                    (content, content, mem_id))
            self._conn.commit()
        return cur.rowcount > 0

    async def delete_long(self, mem_id: int) -> bool:
        """删除一条长期记忆（连同 FTS 索引行）。"""
        return await self._call(self._delete_long_sync, mem_id)

    def _delete_long_sync(self, mem_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM long_memory WHERE id=?", (mem_id,))
            self._conn.execute("DELETE FROM memory_fts WHERE rowid=?", (mem_id,))
            self._conn.commit()
        return cur.rowcount > 0

    async def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """列出会话摘要（后台「记忆管理」页用）。"""
        return await self._call(self._list_sessions_sync, limit)

    def _list_sessions_sync(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY ended_ts DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    async def delete_session(self, session_id: int) -> bool:
        return await self._call(self._delete_session_sync, session_id)

    def _delete_session_sync(self, session_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            self._conn.commit()
        return cur.rowcount > 0

    async def list_messages(self, limit: int = 100, user: str = "") -> list[dict[str, Any]]:
        """列出原始对话流水（后台「记忆管理」页用，倒序）。"""
        return await self._call(self._list_messages_sync, limit, user)

    def _list_messages_sync(self, limit: int, user: str) -> list[dict[str, Any]]:
        with self._lock:
            if user:
                rows = self._conn.execute(
                    "SELECT * FROM messages WHERE user=? ORDER BY id DESC LIMIT ?",
                    (user, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM messages ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    async def delete_message(self, msg_id: int) -> bool:
        return await self._call(self._delete_message_sync, msg_id)

    def _delete_message_sync(self, msg_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM messages WHERE id=?", (msg_id,))
            self._conn.commit()
        return cur.rowcount > 0

    async def list_profiles(self) -> list[dict[str, Any]]:
        """列出全部用户画像（后台展示）。"""
        return await self._call(self._list_profiles_sync)

    def _list_profiles_sync(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM profile ORDER BY updated_ts DESC").fetchall()
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["facts"] = json.loads(item.get("facts") or "[]")
            except json.JSONDecodeError:
                item["facts"] = []
            out.append(item)
        return out

    async def delete_profile(self, user: str) -> bool:
        return await self._call(self._delete_profile_sync, user)

    def _delete_profile_sync(self, user: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM profile WHERE user=?", (user,))
            self._conn.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------ profile
    async def get_profile(self, user: str = "") -> dict[str, Any]:
        """读用户画像；不存在时返回空画像。"""
        return await self._call(self._get_profile_sync, user or DEFAULT_USER)

    def _get_profile_sync(self, user: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT nick, facts FROM profile WHERE user=?", (user,)).fetchone()
        if row is None:
            return {"nick": "", "facts": []}
        try:
            facts = json.loads(row["facts"] or "[]")
        except json.JSONDecodeError:
            facts = []
        return {"nick": row["nick"] or "", "facts": facts}

    async def upsert_profile(self, user: str = "", *, nick: str = "",
                             facts: Iterable[str] = ()) -> None:
        """合并式更新画像：facts 去重合并，nick 非空才覆盖。"""
        facts = [f for f in facts if f]
        await self._call(self._upsert_profile_sync, user or DEFAULT_USER, nick, facts)

    def _upsert_profile_sync(self, user: str, nick: str, facts: list[str]) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT nick, facts FROM profile WHERE user=?", (user,)).fetchone()
            old_nick, old_facts = "", []
            if row is not None:
                old_nick = row["nick"] or ""
                try:
                    old_facts = json.loads(row["facts"] or "[]")
                except json.JSONDecodeError:
                    old_facts = []
            merged = list(dict.fromkeys(old_facts + facts))[:20]   # 去重并限量
            self._conn.execute(
                "INSERT INTO profile(user, nick, facts, updated_ts) VALUES(?,?,?,?)"
                " ON CONFLICT(user) DO UPDATE SET nick=excluded.nick,"
                " facts=excluded.facts, updated_ts=excluded.updated_ts",
                (user, nick or old_nick, json.dumps(merged, ensure_ascii=False),
                 time.time()))
            self._conn.commit()

    # ------------------------------------------------------------------ 结构化事实
    # Mem0 风格：「写前查重 + 冲突消解」——同一 (subject, predicate) 槽位视为同一件事，
    # 新值覆盖旧值，避免自由文本一条条堆叠造成矛盾/重复。
    async def upsert_fact(self, subject: str, predicate: str, obj: str, *,
                          user: str = "", importance: float = 0.5,
                          fact: str = "") -> str:
        """写入一条结构化事实；同槽位则覆盖合并，返回 "added" 或 "updated"。"""
        return await self._call(self._upsert_fact_sync, subject, predicate, obj,
                                user or "", float(importance), fact)

    def _upsert_fact_sync(self, subject: str, predicate: str, obj: str, user: str,
                          importance: float, fact: str) -> str:
        subject, predicate, obj = subject.strip(), predicate.strip(), obj.strip()
        now = time.time()
        text = fact.strip() or f"{subject}{predicate}{obj}"
        with self._lock:
            row = self._conn.execute(
                "SELECT id, importance FROM entity_facts"
                " WHERE user=? AND lower(subject)=lower(?) AND lower(predicate)=lower(?)",
                (user, subject, predicate)).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO entity_facts(user, subject, predicate, object, fact,"
                    " importance, first_seen, last_seen, updated_count)"
                    " VALUES(?,?,?,?,?,?,?,?,1)",
                    (user, subject, predicate, obj, text, importance, now, now))
                self._conn.commit()
                return "added"
            # 同槽位：新值覆盖旧值，重要性取较高者，刷新 last_seen 并累加更新次数
            self._conn.execute(
                "UPDATE entity_facts SET object=?, fact=?, importance=?, last_seen=?,"
                " updated_count=updated_count+1 WHERE id=?",
                (obj, text, max(importance, row["importance"]), now, row["id"]))
            self._conn.commit()
        return "updated"

    async def list_facts(self, user: str = "", limit: int = 100) -> list[dict[str, Any]]:
        """列出结构化事实（后台展示 / 测试用）。"""
        return await self._call(self._list_facts_sync, user, limit)

    def _list_facts_sync(self, user: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            if user:
                rows = self._conn.execute(
                    "SELECT * FROM entity_facts WHERE user IN ('', ?)"
                    " ORDER BY importance DESC, last_seen DESC LIMIT ?",
                    (user, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM entity_facts ORDER BY importance DESC, last_seen DESC"
                    " LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    async def search_facts(self, query: str, *, top_k: int = 4,
                           half_life_days: float = 60.0, user: str = "") -> list[str]:
        """检索结构化事实：2-gram 相关度 × importance × 衰减（中文友好、无需分词）。"""
        return await self._call(self._search_facts_sync, query, top_k,
                                half_life_days, user)

    def _search_facts_sync(self, query: str, top_k: int, half_life_days: float,
                           user: str) -> list[str]:
        now = time.time()
        half_life = max(half_life_days, 1.0) * 86400.0
        with self._lock:
            if user:
                rows = self._conn.execute(
                    "SELECT fact, importance, last_seen FROM entity_facts"
                    " WHERE user IN ('', ?)", (user,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT fact, importance, last_seen FROM entity_facts").fetchall()
        scored: list[tuple[float, str]] = []
        for r in rows:
            relevance = _relevance(query, r["fact"])
            if relevance <= 0:
                continue
            decay = math.exp(-max(now - r["last_seen"], 0.0) / half_life)
            scored.append((relevance * r["importance"] * decay, r["fact"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

    # ------------------------------------------------------------------ 核心记忆
    # MemGPT/Letta core memory 风格：少量「始终注入提示词」的高价值事实，由 Agent 主动提升。
    async def pin_core(self, content: str, *, user: str = "", importance: float = 0.8) -> int:
        """把一条内容提升为核心记忆（内容重复则幂等，返回其 id）。"""
        return await self._call(self._pin_core_sync, content, user or "", float(importance))

    def _pin_core_sync(self, content: str, user: str, importance: float) -> int:
        content = content.strip()
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM core_facts WHERE user=? AND content=?",
                (user, content)).fetchone()
            if row is not None:
                return int(row["id"])              # 已存在 → 幂等，不重复写
            cur = self._conn.execute(
                "INSERT INTO core_facts(user, content, importance, created_ts)"
                " VALUES(?,?,?,?)", (user, content, importance, time.time()))
            self._conn.commit()
        return int(cur.lastrowid)

    async def unpin_core(self, content: str, *, user: str = "") -> bool:
        """取消一条核心记忆。"""
        return await self._call(self._unpin_core_sync, content.strip(), user or "")

    def _unpin_core_sync(self, content: str, user: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM core_facts WHERE user=? AND content=?", (user, content))
            self._conn.commit()
        return cur.rowcount > 0

    async def list_core(self, *, user: str = "", limit: int = 8) -> list[str]:
        """取核心记忆内容（按重要性、时间排序），供决策提示词常驻注入。"""
        return await self._call(self._list_core_sync, user, limit)

    def _list_core_sync(self, user: str, limit: int) -> list[str]:
        with self._lock:
            if user:
                rows = self._conn.execute(
                    "SELECT content FROM core_facts WHERE user IN ('', ?)"
                    " ORDER BY importance DESC, created_ts ASC LIMIT ?",
                    (user, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT content FROM core_facts"
                    " ORDER BY importance DESC, created_ts ASC LIMIT ?",
                    (limit,)).fetchall()
        return [r["content"] for r in rows]

    # ------------------------------------------------------------------ 辅助
    def stats(self) -> dict[str, int]:
        """各表行数（人工核对用）。"""
        with self._lock:
            out = {}
            for table in ("messages", "sessions", "long_memory", "profile",
                          "entity_facts", "core_facts"):
                out[table] = self._conn.execute(
                    f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        return out


_GRAM_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def _make_match_expr(query: str, max_grams: int = 8) -> str:
    """把自然语言查询转成 FTS5 trigram 能用的 MATCH 表达式。

    trigram 分词只对「≥3 个字符」的短语生效，所以把每个词块滑窗切成 3-gram 再 OR：
        之前说过我喜欢吃什么 → "我之前" OR "之前说" OR "前说过" ...
    这样无需中文分词器也能实现子串召回。
    """
    chunks = _GRAM_RE.findall((query or "").lower())
    grams: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if len(chunk) < 3:
            continue
        for i in range(len(chunk) - 2):
            gram = chunk[i:i + 3]
            if gram not in seen:
                seen.add(gram)
                grams.append(gram)
            if len(grams) >= max_grams:
                break
        if len(grams) >= max_grams:
            break
    if not grams:
        return ""
    # 每个 gram 用双引号包裹成短语，避免被当成 FTS 语法
    return " OR ".join(f'"{g}"' for g in grams)


def _normalize_relevance(pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """把「越大越相关」的原始分归一到 (0,1]，好与 importance、衰减同尺度相乘。

    同一次查询内所有候选走同一条路径（要么都来自 FTS 的 bm25，要么都来自 LIKE），
    所以按候选内最大值归一化是自洽的。
    """
    top = max((v for _, v in pairs), default=0.0)
    if top <= 0:
        return [(rid, 1.0) for rid, _ in pairs]
    return [(rid, v / top) for rid, v in pairs]


def _char_grams(text: str, n: int = 2) -> set[str]:
    """把文本切成 n-gram 字符集合（按词块滑窗），用于无分词器的中文相关度估算。"""
    grams: set[str] = set()
    for chunk in _GRAM_RE.findall((text or "").lower()):
        if len(chunk) <= n:
            grams.add(chunk)
            continue
        for i in range(len(chunk) - n + 1):
            grams.add(chunk[i:i + n])
    return grams


def _relevance(query: str, text: str) -> float:
    """相关度 = 命中的查询 n-gram 比例（0~1），中文友好且零依赖。"""
    q = _char_grams(query)
    if not q:
        return 0.0
    return len(q & _char_grams(text)) / len(q)