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

    async def summarize(self, msgs: list[Message], user: str = "",
                        *, write: bool = True) -> str:
        """把一段对话压成摘要，返回摘要文本（阶段三：也被压缩流程复用）。

        Args:
            write: True=照常写 sessions 与 long_memory；False=只做内存侧压缩
                   （沙盒会话 persist=false 时用，不落任何库）。

        说明：本方法被 Mind 的「压缩」路径直接 await（需要拿到摘要文本重建上下文），
        也被后台 _run 复用（会话边界触发）。失败一律只记日志并返回空串。
        """
        if not msgs:
            return ""
        dialogue = _render(msgs)
        try:
            resp = await self.gateway.chat([
                ChatMessage(role="system", content=prompts.SUMMARY_SYSTEM),
                ChatMessage(role="user", content=dialogue),
            ], temperature=0.3, task="summary")
            summary = (resp.content or "").strip()
        except LLMError as e:
            logger.warning("会话摘要失败：%s", e)
            return ""
        if not summary or not write:
            return summary
        user_key = user or DEFAULT_USER
        await self.store.add_session(summary, user=user_key,
                                     started_ts=msgs[0].ts or 0.0,
                                     ended_ts=msgs[-1].ts or 0.0)
        await self.store.add_long(summary, source="chat_summary",
                                  importance=0.5, user=user_key)
        return summary

    async def organize(self, msgs: list[Message], user: str = "",
                       *, write: bool = True) -> str:
        """一次「记忆整理」：滚动摘要 + 画像抽取 + 结构化事实反思。

        阶段三的压缩流程直接 await 本方法（需要摘要文本重建上下文）；
        沙盒会话 write=False → 只产出摘要文本供内存侧压缩，不写任何库。
        返回摘要文本（失败/为空时返回空串）。
        """
        summary = await self.summarize(msgs, user, write=write)
        if not summary or not write:
            return summary
        user_key = user or DEFAULT_USER
        dialogue = _render(msgs)
        await self._extract_profile(user_key, dialogue)
        await self._extract_facts(user_key, dialogue)
        return summary

    async def _extract_profile(self, user_key: str, dialogue: str) -> None:
        """画像抽取（固定 JSON）：把这段对话里的稳定事实 upsert 进画像。"""
        try:
            data = await structured_chat(self.gateway, [
                ChatMessage(role="system", content=prompts.PROFILE_SYSTEM),
                ChatMessage(role="user", content=dialogue),
            ], retry=0, temperature=0.2, task="profile")
        except (StructuredError, LLMError) as e:
            logger.info("画像抽取跳过：%s", e)
            return
        nick = str(data.get("nick") or "").strip()
        raw_facts = data.get("facts") or []
        facts = [str(f).strip() for f in raw_facts if str(f).strip()][:5]
        if nick or facts:
            await self.store.upsert_profile(user_key, nick=nick, facts=facts)

    async def _extract_facts(self, user_key: str, dialogue: str) -> None:
        """结构化事实反思（Mem0 风格）：抽取 (subject, predicate, object) 并合并进 entity_facts。

        这里等于「只压缩不升华」的补课：会话边界不只写一句摘要，还把可去重的事实沉淀下来，
        下次检索就能按三因子召回，而不是靠自由文本堆叠。
        """
        try:
            data = await structured_chat(self.gateway, [
                ChatMessage(role="system", content=prompts.FACTS_SYSTEM),
                ChatMessage(role="user", content=dialogue),
            ], retry=0, temperature=0.2, task="facts")
        except (StructuredError, LLMError) as e:
            logger.info("事实抽取跳过：%s", e)
            return
        for item in (data.get("facts") or [])[:5]:
            if not isinstance(item, dict):
                continue
            subject = str(item.get("subject") or "对方").strip()
            predicate = str(item.get("predicate") or "").strip()
            obj = str(item.get("object") or "").strip()
            if not predicate or not obj:
                continue                      # 缺主谓/值的事实无意义，丢弃
            try:
                importance = float(item.get("importance") or 0.5)
            except (TypeError, ValueError):
                importance = 0.5
            # 同 (subject, predicate) 槽位在 store 内自动覆盖合并（冲突消解）
            await self.store.upsert_fact(subject, predicate, obj,
                                         user=user_key, importance=importance)

    async def _run(self, msgs: list[Message], user: str) -> None:
        """后台任务：会话边界的摘要 + 画像抽取（任何失败只记日志）。"""
        await self.organize(msgs, user, write=True)

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