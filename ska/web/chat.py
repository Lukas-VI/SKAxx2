"""ChatHub —— 在线聊天会话中枢（阶段三，见方案 §2.1/§2.2）。

它解决「浏览器怎么跟核心对话」这一件事，刻意做成**内核旁挂的搬运工**：

    浏览器 ── POST /api/chat/send ──► ChatHub.submit()
                                        │ 组装 CoreInput(source=webchat_message)
                                        ▼
                                    EventBus.publish ──► 心跳主循环 ──► Mind 五步
                                        ▲
    webchat 渠道 send(CoreOutput) ── ChatHub.push() ──► 会话出站队列
                                        │
    浏览器 ── GET /api/chat/poll?after= ─┘  （1 秒轮询增量拉取）

为什么不在这里推理：一旦在 Web 层动手做过任何决策，就会绕开内核的权限闸、
记忆落库与 MindLine 追踪。所有推理仍由心跳主循环完成，本类只负责「排队 + 取件」。

并发说明：
    Web 请求是经 server.py 的 run_coro 桥回事件循环执行的，因此本类的
    async 方法（submit）可以直接 await 总线，不需要 run_coroutine_threadsafe。
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..kernel.event import TOPIC_INPUT, Event
from ..kernel.packet import CoreInput, CoreOutput

logger = logging.getLogger(__name__)

# 等待回复的兜底超时：超过这么久还没产出输出，就把 pending 复位
# （例如 Mind 判定 silence、或 LLM 调用失败 —— 不能让前端一直转圈）
PENDING_TIMEOUT = 60.0


@dataclass
class ChatSession:
    """一个在线聊天会话（纯内存，进程退出即消失）。

    Attributes:
        id:       会话标识（前端生成或后端 uuid），也是 SessionRegistry 的 key 一段
        title:    可改的标题（未设置时用首条消息截断生成）
        persist:  True=照常落库；False=沙盒（照常推理，但不写任何记忆库）
        identity: 这条会话代表的「聊天用户身份」（决定权限等级与记忆归属）
        messages: 出站/入站消息流水（含 seq，供前端增量拉取）
        seq:      已分配的最大消息序号
        pending:  是否正在等模型回复（前端显示「思考中」）
    """

    id: str
    title: str = ""
    persist: bool = True
    identity: str = ""
    created_ts: float = field(default_factory=time.time)
    last_ts: float = field(default_factory=time.time)
    messages: list[dict[str, Any]] = field(default_factory=list)
    seq: int = 0
    pending: bool = False
    pending_since: float = 0.0
    tokens: int = 0                     # 最近一次上下文占用的 token 数（状态条展示）

    # ------------------------------------------------------------------ 便捷
    def add(self, role: str, content: str) -> dict[str, Any]:
        """追加一条消息并分配序号（前端按 seq 增量取）。"""
        self.seq += 1
        item = {"seq": self.seq, "role": role, "content": content, "ts": time.time()}
        self.messages.append(item)
        self.last_ts = item["ts"]
        return item

    def info(self) -> dict[str, Any]:
        """会话元信息（列表 / 状态条用）。"""
        return {
            "id": self.id,
            "title": self.title or ("会话 " + self.id[:6]),
            "persist": self.persist,
            "identity": self.identity,
            "created_ts": self.created_ts,
            "last_ts": self.last_ts,
            "count": len(self.messages),
            "last_seq": self.seq,
            "pending": self.pending,
            "tokens": self.tokens,
            "sandbox": not self.persist,
        }


class ChatHub:
    """在线会话登记表 + 出站队列（供 webchat 渠道与聊天 API 共用）。"""

    def __init__(self, bus: Any = None, *, default_identity: str = "owner",
                 default_persist: bool = True) -> None:
        self.bus = bus                       # kernel.bus.EventBus（main.py 注入）
        self.default_identity = default_identity or "owner"
        self.default_persist = bool(default_persist)
        self._sessions: dict[str, ChatSession] = {}

    # ------------------------------------------------------------------ 会话 CRUD
    def create(self, *, title: str = "", persist: bool | None = None,
               identity: str = "", session_id: str = "") -> ChatSession:
        """新建会话。未给 id 时自动生成（uuid 前 12 位，够短也够唯一）。"""
        sid = session_id.strip() or uuid.uuid4().hex[:12]
        if sid in self._sessions:
            raise ValueError(f"会话已存在：{sid}")
        session = ChatSession(
            id=sid,
            title=title.strip(),
            persist=self.default_persist if persist is None else bool(persist),
            identity=(identity.strip() or self.default_identity),
        )
        self._sessions[sid] = session
        logger.info("在线会话创建：%s（persist=%s identity=%s）",
                    sid, session.persist, session.identity)
        return session

    def get(self, session_id: str) -> ChatSession | None:
        return self._sessions.get(session_id)

    def list(self) -> list[dict[str, Any]]:
        """按最近活跃倒序列出会话（前端标签栏顺序）。"""
        return [s.info() for s in
                sorted(self._sessions.values(), key=lambda s: s.last_ts, reverse=True)]

    def update(self, session_id: str, *, title: str | None = None,
               persist: bool | None = None, identity: str | None = None) -> dict[str, Any]:
        """改标题 / 落库开关 / 身份（身份只影响之后的消息，不动已有窗口）。"""
        session = self._require(session_id)
        if title is not None:
            session.title = str(title).strip()
        if persist is not None:
            session.persist = bool(persist)
        if identity is not None and str(identity).strip():
            session.identity = str(identity).strip()
        return session.info()

    def delete(self, session_id: str) -> bool:
        """关闭会话（内存即释放；已落库的对话流水不受影响）。"""
        return self._sessions.pop(session_id, None) is not None

    # ------------------------------------------------------------------ 入站
    async def submit(self, session_id: str, text: str,
                     user_name: str = "") -> dict[str, Any]:
        """把一条用户输入投进核心推理链。

        返回投递后的会话快照（含刚追加的 user 消息序号），供前端立即回显。
        """
        session = self._require(session_id)
        content = (text or "").strip()
        if not content:
            raise ValueError("消息内容不能为空")
        item = session.add("user", content)
        session.pending = True
        session.pending_since = time.time()
        if not session.title:                       # 首条消息自动当标题
            session.title = content[:24]

        packet = CoreInput(
            source="webchat_message",
            content=content,
            user=session.identity,
            user_name=user_name or session.identity,
            channel="webchat",
            session=session.id,
            sandbox=not session.persist,            # 沙盒会话：不落库
        )
        if self.bus is None:
            logger.warning("ChatHub 未接入总线，消息被丢弃：%s", session_id)
            raise RuntimeError("核心总线未就绪")
        # 走总线 → 心跳主循环 drain → Mind 五步（白送记忆/权限/MindLine）
        # 这里与 io/base.py 的 Channel.publish_input 同款：入队 + 唤醒主循环。
        await self.bus.publish(Event(topic=TOPIC_INPUT, payload=packet))
        return {"session": session.info(), "message": item}

    # ------------------------------------------------------------------ 出站
    def push(self, session_id: str, out: CoreOutput) -> bool:
        """webchat 渠道 send() 调用：把模型回复放进对应会话队列。"""
        session = self._sessions.get(session_id) if session_id else None
        if session is None:
            # 输出没带会话（或会话已关闭）：落到最近活跃的会话，至少不丢内容
            session = self._latest()
        if session is None:
            logger.warning("没有可投递的在线会话，回复被丢弃")
            return False
        session.add("assistant", out.content)
        session.pending = False
        session.pending_since = 0.0
        return True

    def on_trace(self, payload: Any) -> None:
        """订阅 TOPIC_TRACE 的回调：把上下文占用 token 回填到会话状态条。

        只认「带 tokens 且 belong 到已知会话」的事件，其余忽略。
        """
        sid = str(getattr(payload, "session", "") or "")
        session = self._sessions.get(sid)
        if session is None:
            return
        tokens = int(getattr(payload, "tokens", 0) or 0)
        if tokens > 0:
            session.tokens = tokens

    # ------------------------------------------------------------------ 拉取
    def poll(self, session_id: str, after: int = 0) -> dict[str, Any]:
        """增量拉取（msgs.seq > after）。前端 1 秒轮询一次。"""
        session = self._require(session_id)
        self._expire_pending(session)
        fresh = [m for m in session.messages if int(m["seq"]) > int(after)]
        return {"session": session.info(), "messages": fresh,
                "last_seq": session.seq}

    def history(self, session_id: str) -> dict[str, Any]:
        """全量回看（打开旧会话时用）。"""
        session = self._require(session_id)
        return {"session": session.info(), "messages": list(session.messages)}

    # ------------------------------------------------------------------ 内部
    def _require(self, session_id: str) -> ChatSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f"会话不存在：{session_id}")
        return session

    def _latest(self) -> ChatSession | None:
        if not self._sessions:
            return None
        return max(self._sessions.values(), key=lambda s: s.last_ts)

    @staticmethod
    def _expire_pending(session: ChatSession) -> None:
        """等太久还没回复（沉默/调用失败）→ 复位 pending，别让前端一直转圈。"""
        if session.pending and time.time() - session.pending_since > PENDING_TIMEOUT:
            session.pending = False
            session.pending_since = 0.0
