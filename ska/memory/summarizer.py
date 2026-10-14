"""会话摘要 + 画像抽取（后台低频任务）。

会话边界（消息数 / 空闲 / 跨天）触发时由 Mind 调用 schedule()：
    * 立刻返回，摘要工作作为后台 asyncio.Task 执行，不阻塞本轮对话；
    * 任务流程：对话 → 150-200 字摘要 → 写 sessions 与 long_memory；
              同一批对话 → 固定 JSON 抽画像 → upsert profile。

后台任务集合持有引用，避免被 GC 提前回收（asyncio 的常见坑）。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..kernel.context import Message
from ..llm.gateway import LLMError
from ..llm.models import ChatMessage
from ..llm.structured import StructuredError, structured_chat
from ..loop import prompts
from .store import DEFAULT_USER

logger = logging.getLogger(__name__)


class Summarizer:
    """会话摘要与画像抽取。"""

    def __init__(self, gateway: Any, store: Any) -> None:
        self.gateway = gateway
        self.store = store
        self._tasks: set[asyncio.Task] = set()

    def schedule(self, msgs: list[Message], user: str = "") -> None:
        """把一批消息排进后台摘要任务（fire-and-forget）。"""
        if not msgs:
            return
        task = asyncio.create_task(self._run(list(msgs), user))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, msgs: list[Message], user: str) -> None:
        """执行一次摘要 + 画像抽取。任何失败都只记日志，不影响主流程。"""
        started = msgs[0].ts or 0.0
        ended = msgs[-1].ts or 0.0
        dialogue = _render(msgs)
        user_key = user or DEFAULT_USER

        # 1) 会话摘要
        try:
            resp = await self.gateway.chat([
                ChatMessage(role="system", content=prompts.SUMMARY_SYSTEM),
                ChatMessage(role="user", content=dialogue),
            ], temperature=0.3)
            summary = (resp.content or "").strip()
        except LLMError as e:
            logger.warning("会话摘要失败：%s", e)
            return
        if not summary:
            return
        await self.store.add_session(summary, user=user_key,
                                     started_ts=started, ended_ts=ended)
        await self.store.add_long(summary, source="chat_summary",
                                  importance=0.5, user=user_key)

        # 2) 画像抽取（固定 JSON）
        try:
            data = await structured_chat(self.gateway, [
                ChatMessage(role="system", content=prompts.PROFILE_SYSTEM),
                ChatMessage(role="user", content=dialogue),
            ], retry=0, temperature=0.2)
        except (StructuredError, LLMError) as e:
            logger.info("画像抽取跳过：%s", e)
            return
        nick = str(data.get("nick") or "").strip()
        raw_facts = data.get("facts") or []
        facts = [str(f).strip() for f in raw_facts if str(f).strip()][:5]
        if nick or facts:
            await self.store.upsert_profile(user_key, nick=nick, facts=facts)

    async def close(self) -> None:
        """等待未完成的后台任务（退出时调用）。"""
        pending = [t for t in self._tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


def _render(msgs: list[Message]) -> str:
    """把消息渲染成给摘要/画像用的对话文本。"""
    lines = []
    for m in msgs:
        who = "对方" if m.role == "user" else ("你" if m.role == "assistant" else m.role)
        lines.append(f"{who}：{m.content}")
    return "\n".join(lines)