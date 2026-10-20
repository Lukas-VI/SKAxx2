"""后台管理数据存储（SQLite，独立库 data/admin.db）。

与 MemoryStore（data/ska.db，存「对话与记忆」）刻意分开：
    * 关注点不同：这里是「账号 / 配额 / 元数据 / 白名单」，不是对话内容；
    * 生命周期不同：清空记忆不应影响管理员口令与用量统计。

表结构（见方案 §5.6）：
    admin_users       后台账号（口令 PBKDF2 哈希 + salt + 角色）
    sessions          登录会话令牌（带过期时间）
    settings          运行时可改项（key → JSON 文本；覆盖 config.yaml）
    usage             LLM token 用量流水（按 provider/model/task 打标签）
    chat_users        出现过的聊天用户及其等级/拉黑状态（权限用）
    publish_targets   「机器人可主动发言」的群/用户白名单
    tool_permissions  工具级开关与最小可用等级

并发模型与 MemoryStore 完全一致：单连接 + threading.Lock，
所有阻塞操作经 asyncio.to_thread 丢线程执行，绝不阻塞事件循环。
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS admin_users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT NOT NULL UNIQUE,
    pass_hash     TEXT NOT NULL,
    salt          TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'owner',
    created_ts    REAL NOT NULL,
    last_login_ts REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    username   TEXT NOT NULL,
    created_ts REAL NOT NULL,
    expires_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_ts);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                REAL NOT NULL,
    provider          TEXT DEFAULT '',
    model             TEXT DEFAULT '',
    task              TEXT DEFAULT '',
    prompt_tokens     INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens      INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts);

CREATE TABLE IF NOT EXISTS chat_users (
    user_id     TEXT NOT NULL,
    channel     TEXT NOT NULL DEFAULT '',
    name        TEXT DEFAULT '',
    level       TEXT NOT NULL DEFAULT 'member',
    blacklisted INTEGER NOT NULL DEFAULT 0,
    note        TEXT DEFAULT '',
    first_ts    REAL NOT NULL,
    last_ts     REAL NOT NULL,
    msg_count   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (channel, user_id)
);

CREATE TABLE IF NOT EXISTS publish_targets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    channel     TEXT NOT NULL DEFAULT 'qq',
    target_type TEXT NOT NULL DEFAULT 'group',   -- group | user
    target_id   TEXT NOT NULL,
    name        TEXT DEFAULT '',
    enabled     INTEGER NOT NULL DEFAULT 1,
    note        TEXT DEFAULT '',
    created_ts  REAL NOT NULL,
    UNIQUE (channel, target_type, target_id)
);

CREATE TABLE IF NOT EXISTS tool_permissions (
    tool      TEXT PRIMARY KEY,
    enabled   INTEGER NOT NULL DEFAULT 1,
    min_level TEXT NOT NULL DEFAULT 'member',
    note      TEXT DEFAULT ''
);

-- 阶段三 MindLine（决策 E4：trace 作为 log 的一部分落库）
-- seq 由 TraceLog 全局递增分配；session 用于多开会话过滤；sandbox 标注沙盒会话。
CREATE TABLE IF NOT EXISTS traces (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id  TEXT DEFAULT '',
    seq       INTEGER NOT NULL DEFAULT 0,
    ts        REAL NOT NULL,
    session   TEXT DEFAULT '',
    lane      TEXT DEFAULT '',
    kind      TEXT DEFAULT '',
    title     TEXT DEFAULT '',
    detail    TEXT DEFAULT '',
    tokens    INTEGER DEFAULT 0,
    sandbox   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_traces_seq ON traces(seq);
"""


class AdminStore:
    """后台管理库。所有对外方法均为 async（内部 to_thread 执行阻塞 sqlite）。"""

    def __init__(self, db_path: str | Path = "data/admin.db") -> None:
        self.path = Path(db_path)
        if not self.path.is_absolute():
            from ..kernel.config import package_root
            self.path = package_root() / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("管理库就绪：%s", self.path)

    # ------------------------------------------------------------------ 线程封装
    async def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """把阻塞的 sqlite 操作放到线程池执行（与 MemoryStore 同款做法）。"""
        return await asyncio.to_thread(fn, *args, **kwargs)

    async def close(self) -> None:
        await self._call(self._close_sync)

    def _close_sync(self) -> None:
        with self._lock:
            self._conn.close()

    # ================================================================== settings
    def all_settings_sync(self) -> dict[str, Any]:
        """同步读取全部设置（仅供启动阶段构造 RuntimeSettings 使用，此时无事件循环）。"""
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM settings").fetchall()
        out: dict[str, Any] = {}
        for row in rows:
            try:
                out[row["key"]] = json.loads(row["value"])
            except json.JSONDecodeError:
                out[row["key"]] = row["value"]
        return out

    async def all_settings(self) -> dict[str, Any]:
        return await self._call(self.all_settings_sync)

    async def save_settings(self, data: dict[str, Any]) -> None:
        """整体覆盖式写入设置（key → JSON）。WebUI 保存设置时调用。"""
        await self._call(self._save_settings_sync, data)

    def _save_settings_sync(self, data: dict[str, Any]) -> None:
        with self._lock:
            for key, value in data.items():
                self._conn.execute(
                    "INSERT INTO settings(key, value) VALUES(?,?)"
                    " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(key), json.dumps(value, ensure_ascii=False)))
            self._conn.commit()

    # ================================================================== admin_users
    async def count_admins(self) -> int:
        return await self._call(self._count_admins_sync)

    def _count_admins_sync(self) -> int:
        with self._lock:
            return int(self._conn.execute(
                "SELECT COUNT(*) AS c FROM admin_users").fetchone()["c"])

    async def create_admin(self, username: str, pass_hash: str, salt: str,
                           role: str = "owner") -> int:
        return await self._call(self._create_admin_sync, username, pass_hash, salt, role)

    def _create_admin_sync(self, username: str, pass_hash: str, salt: str, role: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO admin_users(username, pass_hash, salt, role, created_ts)"
                " VALUES(?,?,?,?,?)",
                (username, pass_hash, salt, role, time.time()))
            self._conn.commit()
        return int(cur.lastrowid)

    async def get_admin(self, username: str) -> dict[str, Any] | None:
        return await self._call(self._get_admin_sync, username)

    def _get_admin_sync(self, username: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM admin_users WHERE username=?", (username,)).fetchone()
        return dict(row) if row else None

    async def list_admins(self) -> list[dict[str, Any]]:
        """列出后台账号（隐去口令哈希，只回展示所需字段）。"""
        return await self._call(self._list_admins_sync)

    def _list_admins_sync(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, username, role, created_ts, last_login_ts"
                " FROM admin_users ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    async def update_admin(self, admin_id: int, *, role: str | None = None,
                           pass_hash: str | None = None, salt: str | None = None) -> bool:
        return await self._call(self._update_admin_sync, admin_id, role, pass_hash, salt)

    def _update_admin_sync(self, admin_id: int, role: str | None,
                           pass_hash: str | None, salt: str | None) -> bool:
        fields: list[str] = []
        values: list[Any] = []
        if role is not None:
            fields.append("role=?"); values.append(role)
        if pass_hash is not None and salt is not None:
            fields.append("pass_hash=?"); values.append(pass_hash)
            fields.append("salt=?"); values.append(salt)
        if not fields:
            return False
        values.append(admin_id)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE admin_users SET {', '.join(fields)} WHERE id=?", values)
            self._conn.commit()
        return cur.rowcount > 0

    async def delete_admin(self, admin_id: int) -> bool:
        return await self._call(self._delete_admin_sync, admin_id)

    def _delete_admin_sync(self, admin_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM admin_users WHERE id=?", (admin_id,))
            self._conn.commit()
        return cur.rowcount > 0

    async def touch_login(self, username: str) -> None:
        await self._call(self._touch_login_sync, username)

    def _touch_login_sync(self, username: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE admin_users SET last_login_ts=? WHERE username=?",
                               (time.time(), username))
            self._conn.commit()

    # ================================================================== sessions
    async def create_session(self, token: str, username: str, expires_ts: float) -> None:
        await self._call(self._create_session_sync, token, username, expires_ts)

    def _create_session_sync(self, token: str, username: str, expires_ts: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions(token, username, created_ts, expires_ts)"
                " VALUES(?,?,?,?)", (token, username, time.time(), expires_ts))
            self._conn.commit()

    async def get_session(self, token: str) -> dict[str, Any] | None:
        """取有效会话；过期则顺手删掉并返回 None。"""
        return await self._call(self._get_session_sync, token)

    def _get_session_sync(self, token: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE token=?", (token,)).fetchone()
            if row is None:
                return None
            if row["expires_ts"] < time.time():          # 过期即清理
                self._conn.execute("DELETE FROM sessions WHERE token=?", (token,))
                self._conn.commit()
                return None
        return dict(row)

    async def delete_session(self, token: str) -> None:
        await self._call(self._delete_session_sync, token)

    def _delete_session_sync(self, token: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE token=?", (token,))
            self._conn.commit()

    # ================================================================== usage
    async def record_usage(self, provider: str, model: str, task: str,
                           prompt_tokens: int = 0, completion_tokens: int = 0,
                           total_tokens: int = 0) -> None:
        await self._call(self._record_usage_sync, provider, model, task,
                         int(prompt_tokens), int(completion_tokens), int(total_tokens))

    def _record_usage_sync(self, provider: str, model: str, task: str,
                           prompt_tokens: int, completion_tokens: int,
                           total_tokens: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage(ts, provider, model, task, prompt_tokens,"
                " completion_tokens, total_tokens) VALUES(?,?,?,?,?,?,?)",
                (time.time(), provider, model, task,
                 prompt_tokens, completion_tokens, total_tokens))
            self._conn.commit()

    async def usage_summary(self, days: int = 7, group_by: str = "provider") -> dict[str, Any]:
        """按维度聚合用量。group_by ∈ {provider, model, task, day}。"""
        return await self._call(self._usage_summary_sync, days, group_by)

    def _usage_summary_sync(self, days: int, group_by: str) -> dict[str, Any]:
        column = {"provider": "provider", "model": "model", "task": "task",
                  "day": "date(ts, 'unixepoch', 'localtime')"}.get(group_by, "provider")
        since = time.time() - days * 86400 if days > 0 else 0
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {column} AS grp, COUNT(*) AS calls,"
                " SUM(prompt_tokens) AS prompt_tokens,"
                " SUM(completion_tokens) AS completion_tokens,"
                " SUM(total_tokens) AS total_tokens"
                " FROM usage WHERE ts >= ? GROUP BY grp ORDER BY total_tokens DESC",
                (since,)).fetchall()
            total = self._conn.execute(
                "SELECT COUNT(*) AS calls, COALESCE(SUM(total_tokens),0) AS total_tokens"
                " FROM usage WHERE ts >= ?", (since,)).fetchone()
        return {
            "group_by": group_by, "days": days,
            "totals": {"calls": int(total["calls"]), "total_tokens": int(total["total_tokens"])},
            "rows": [{"group": r["grp"], "calls": int(r["calls"] or 0),
                      "prompt_tokens": int(r["prompt_tokens"] or 0),
                      "completion_tokens": int(r["completion_tokens"] or 0),
                      "total_tokens": int(r["total_tokens"] or 0)} for r in rows],
        }

    async def usage_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self._call(self._usage_recent_sync, limit)

    def _usage_recent_sync(self, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM usage ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ================================================================== chat_users
    async def touch_chat_user(self, channel: str, user_id: str, name: str = "") -> None:
        """记录/更新一个聊天用户（首次插入，之后累加消息数并刷新昵称与时间）。"""
        await self._call(self._touch_chat_user_sync, channel, user_id, name)

    def _touch_chat_user_sync(self, channel: str, user_id: str, name: str) -> None:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT name FROM chat_users WHERE channel=? AND user_id=?",
                (channel, user_id)).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO chat_users(user_id, channel, name, level, blacklisted,"
                    " first_ts, last_ts, msg_count) VALUES(?,?,?,'member',0,?,?,1)",
                    (user_id, channel, name, now, now))
            else:
                self._conn.execute(
                    "UPDATE chat_users SET last_ts=?, msg_count=msg_count+1,"
                    " name=CASE WHEN ?<>'' THEN ? ELSE name END"
                    " WHERE channel=? AND user_id=?",
                    (now, name, name, channel, user_id))
            self._conn.commit()

    async def list_chat_users(self) -> list[dict[str, Any]]:
        return await self._call(self._list_chat_users_sync)

    def _list_chat_users_sync(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chat_users ORDER BY last_ts DESC").fetchall()
        return [dict(r) for r in rows]

    async def get_chat_user(self, channel: str, user_id: str) -> dict[str, Any] | None:
        return await self._call(self._get_chat_user_sync, channel, user_id)

    def _get_chat_user_sync(self, channel: str, user_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM chat_users WHERE channel=? AND user_id=?",
                (channel, user_id)).fetchone()
        return dict(row) if row else None

    async def update_chat_user(self, channel: str, user_id: str, *,
                               level: str | None = None,
                               blacklisted: bool | None = None,
                               note: str | None = None) -> bool:
        return await self._call(self._update_chat_user_sync, channel, user_id,
                                level, blacklisted, note)

    def _update_chat_user_sync(self, channel: str, user_id: str, level: str | None,
                               blacklisted: bool | None, note: str | None) -> bool:
        fields: list[str] = []
        values: list[Any] = []
        if level is not None:
            fields.append("level=?"); values.append(level)
        if blacklisted is not None:
            fields.append("blacklisted=?"); values.append(1 if blacklisted else 0)
        if note is not None:
            fields.append("note=?"); values.append(note)
        if not fields:
            return False
        values.extend([channel, user_id])
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE chat_users SET {', '.join(fields)}"
                " WHERE channel=? AND user_id=?", values)
            self._conn.commit()
        return cur.rowcount > 0

    # ================================================================== publish_targets
    async def list_targets(self) -> list[dict[str, Any]]:
        return await self._call(self._list_targets_sync)

    def _list_targets_sync(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM publish_targets ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    async def add_target(self, channel: str, target_type: str, target_id: str,
                         name: str = "", enabled: bool = True, note: str = "") -> int:
        return await self._call(self._add_target_sync, channel, target_type,
                                target_id, name, enabled, note)

    def _add_target_sync(self, channel: str, target_type: str, target_id: str,
                         name: str, enabled: bool, note: str) -> int:
        now = time.time()
        with self._lock:
            # 同目标重复添加时更新而不是报错（白名单天然幂等）
            self._conn.execute(
                "INSERT INTO publish_targets(channel, target_type, target_id, name,"
                " enabled, note, created_ts) VALUES(?,?,?,?,?,?,?)"
                " ON CONFLICT(channel, target_type, target_id) DO UPDATE SET"
                " name=excluded.name, enabled=excluded.enabled, note=excluded.note",
                (channel, target_type, target_id, name, 1 if enabled else 0, note, now))
            row = self._conn.execute(
                "SELECT id FROM publish_targets WHERE channel=? AND target_type=?"
                " AND target_id=?", (channel, target_type, target_id)).fetchone()
            self._conn.commit()
        return int(row["id"])

    async def update_target(self, target_id_row: int, *, enabled: bool | None = None,
                            name: str | None = None, note: str | None = None) -> bool:
        return await self._call(self._update_target_sync, target_id_row, enabled, name, note)

    def _update_target_sync(self, target_id_row: int, enabled: bool | None,
                            name: str | None, note: str | None) -> bool:
        fields: list[str] = []
        values: list[Any] = []
        if enabled is not None:
            fields.append("enabled=?"); values.append(1 if enabled else 0)
        if name is not None:
            fields.append("name=?"); values.append(name)
        if note is not None:
            fields.append("note=?"); values.append(note)
        if not fields:
            return False
        values.append(target_id_row)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE publish_targets SET {', '.join(fields)} WHERE id=?", values)
            self._conn.commit()
        return cur.rowcount > 0

    async def delete_target(self, target_id_row: int) -> bool:
        return await self._call(self._delete_target_sync, target_id_row)

    def _delete_target_sync(self, target_id_row: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM publish_targets WHERE id=?",
                                     (target_id_row,))
            self._conn.commit()
        return cur.rowcount > 0

    # ================================================================== tool_permissions
    async def list_tool_permissions(self) -> list[dict[str, Any]]:
        return await self._call(self._list_tool_permissions_sync)

    def _list_tool_permissions_sync(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM tool_permissions").fetchall()
        return [dict(r) for r in rows]

    async def upsert_tool_permission(self, tool: str, enabled: bool, min_level: str,
                                     note: str = "") -> None:
        await self._call(self._upsert_tool_permission_sync, tool, enabled, min_level, note)

    def _upsert_tool_permission_sync(self, tool: str, enabled: bool, min_level: str,
                                     note: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO tool_permissions(tool, enabled, min_level, note)"
                " VALUES(?,?,?,?) ON CONFLICT(tool) DO UPDATE SET"
                " enabled=excluded.enabled, min_level=excluded.min_level,"
                " note=excluded.note",
                (tool, 1 if enabled else 0, min_level, note))
            self._conn.commit()

    # ================================================================== 概览
    async def stats(self) -> dict[str, int]:
        """各表行数（后台「运行状态」页展示）。"""
        return await self._call(self._stats_sync)

    def _stats_sync(self) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._lock:
            for table in ("admin_users", "sessions", "settings", "usage",
                          "chat_users", "publish_targets", "tool_permissions", "traces"):
                out[table] = int(self._conn.execute(
                    f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"])
        return out

    # ================================================================== 轨迹（阶段三 MindLine）
    async def add_trace(self, *, trace_id: str, seq: int, ts: float, session: str,
                        lane: str, kind: str, title: str = "", detail: str = "",
                        tokens: int = 0, sandbox: bool = False) -> None:
        """写入一条轨迹（TraceLog 调用）。用关键字参数避免十余个位置参数错位。"""
        await self._call(self._add_trace_sync, trace_id, seq, ts, session, lane,
                         kind, title, detail, int(tokens), bool(sandbox))

    def _add_trace_sync(self, trace_id: str, seq: int, ts: float, session: str,
                        lane: str, kind: str, title: str, detail: str,
                        tokens: int, sandbox: bool) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO traces(trace_id, seq, ts, session, lane, kind,"
                " title, detail, tokens, sandbox) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (trace_id, int(seq), float(ts), session, lane, kind, title, detail,
                 int(tokens), 1 if sandbox else 0))
            self._conn.commit()

    async def list_traces(self, after: int = 0, session: str = "", limit: int = 300) -> list[dict[str, Any]]:
        """按 seq 增量查询轨迹（供 /api/trace；库侧兜底，页面通常读内存环）。"""
        return await self._call(self._list_traces_sync, after, session, limit)

    def _list_traces_sync(self, after: int, session: str, limit: int) -> list[dict[str, Any]]:
        sql = "SELECT * FROM traces WHERE seq > ?"
        params: list[Any] = [int(after)]
        if session:
            sql += " AND session = ?"
            params.append(session)
        sql += " ORDER BY seq ASC LIMIT ?"
        params.append(max(int(limit), 1))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    async def clear_traces(self) -> int:
        """清空轨迹表（「运行状态」页的清理按钮用），返回删除条数。"""
        return await self._call(self._clear_traces_sync)

    def _clear_traces_sync(self) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM traces")
            self._conn.commit()
        return cur.rowcount