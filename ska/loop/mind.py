"""Mind —— 自觉决策层。

一次唤醒的时序（对应方案第 2 节 ①~⑧）：
    ① 记忆召回（profile + FTS + 最近摘要，见 memory/recall.py）
    ② 内独白（LLM，只进日志/记忆，不进用户输出）
    ③ 决策（LLM 结构化输出 → Decision：silence / speak / remember / tool）
    ④ 需要工具则交 AgentLoop 执行 think→act→observe
    ⑤ 反思（按长度/重要性触发二次修订，Reflexion）
    ⑥ 组装 MindResult（speak 走 CoreOutput 路由；remember 落库）
    ⑦ 会话边界判定 → 后台摘要 + 画像抽取
    ⑧ 返回，交给心跳循环回到 bus.wait

关键点：老项目 prompt.json 的 self_response「隐式自循环」在这里被显式拆成 ②③④，
不再有「把自己的输出再塞回输入」的递归风险。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from ..kernel.context import Message, SessionContext
from ..kernel.packet import CoreInput, CoreOutput, Decision, MindResult, ToolCall
from ..llm.gateway import LLMError
from ..llm.models import ChatMessage
from ..llm.structured import StructuredError, structured_chat
from ..memory import recall
from . import prompts

logger = logging.getLogger(__name__)

# 反思触发阈值（方案：按重要度/长度触发二次修订）
REFLECT_MIN_LEN = 80
REFLECT_IMPORTANCE = 0.8
# 送给 LLM 的最近对话条数
HISTORY_LIMIT = 12

_VALID_ACTIONS = {"speak", "silence", "remember", "tool"}


def format_history(msgs: list[Message]) -> str:
    """把工作记忆格式化成人类可读的对话文本。"""
    lines: list[str] = []
    for m in msgs:
        if m.role == "user":
            lines.append(f"对方：{m.content}")
        elif m.role == "assistant":
            lines.append(f"你：{m.content}")
        else:
            lines.append(f"（{m.role}）{m.content}")
    return "\n".join(lines) if lines else "（还没有对话）"


class Mind:
    """自觉决策层。"""

    def __init__(
        self,
        gateway: Any,
        agent: Any,
        *,
        store: Any = None,
        summarizer: Any = None,
        cfg: Any = None,
        persona_prompt: str = "",
        policy: Any = None,
    ) -> None:
        self.gateway = gateway
        self.agent = agent            # AgentLoop
        self.store = store            # MemoryStore（可为 None：无记忆也能跑）
        self.summarizer = summarizer  # Summarizer（可为 None）
        self.cfg = cfg
        self.policy = policy          # admin.policy.Policy（阶段二；可为 None=不裁权限）
        self.persona_prompt = persona_prompt      # 保留原始人设，供后台切档时重建
        self.system = prompts.make_system(persona_prompt)
        self.mem_cfg = cfg.memory if cfg else {}
        self.llm_retry = int((cfg.llm if cfg else {}).get("retry", 1))

    # ------------------------------------------------------------------ 人设热更新
    def refresh_persona(self) -> None:
        """后台切换人设档后重建 system prompt（由 AdminService._apply_live 调用）。

        人设文本由 Config.build_system_prompt() 生成；调用方在切档后需先让
        RuntimeSettings 把档位写回 Config，再调用本方法。
        """
        try:
            new_prompt = self.cfg.build_system_prompt() if self.cfg is not None else ""
        except Exception:  # noqa: BLE001 —— 配置异常不应打断既有对话
            logger.exception("重建人设失败，沿用旧 system prompt")
            return
        if new_prompt:
            self.persona_prompt = new_prompt
            self.system = prompts.make_system(new_prompt)
            logger.info("人设已刷新（%d 字）", len(new_prompt))

    # ------------------------------------------------------------------ 主入口
    async def wake(self, ctx: SessionContext, trigger: CoreInput) -> MindResult:
        """处理一次唤醒，返回决策产物。"""
        # ⓪ 权限闸（阶段二）：登记聊天用户；被拉黑者直接忽略（不回话、不进记忆）
        if self.policy is not None and trigger.is_user_message():
            await self.policy.note_user(trigger)
            if not await self.policy.should_respond(trigger):
                return MindResult(decision=Decision(action="silence", reason="用户不可回应"))

        # ① 记忆召回
        memory_block = ""
        if self.store:
            memory_block = await recall.build_memory_block(
                self.store, trigger.content, user=trigger.user, cfg=self.mem_cfg)

        # 记录用户输入到工作记忆（+ 原始流水落库）
        if trigger.is_user_message():
            ctx.add("user", f"{trigger.user_name or trigger.user}: {trigger.content}")
            await self._persist("user", trigger.content, trigger)

        recent = format_history(ctx.recent(HISTORY_LIMIT))
        desc = self._describe(trigger)

        # ② 内独白
        monologue = await self._monologue(desc, recent)

        # ③ 决策
        decision = await self._decide(desc, recent, monologue, memory_block)

        # ④ 工具执行（按用户等级裁剪可用工具；非用户消息不裁剪）
        agent_result = None
        content = decision.content
        if decision.action == "tool":
            allow = await self.policy.tools_for(trigger) if self.policy is not None else None
            agent_result = await self.agent.run(
                self._agent_context(ctx, memory_block), first=decision.tool, allow=allow)
            content = agent_result.final

        # ⑤ 反思（仅对真实用户消息做质量把关，省下主动路径的调用）
        reflection = ""
        if trigger.is_user_message() and content.strip():
            content, reflection = await self._reflect(content, desc, decision)

        # ⑥ 组装输出
        output: CoreOutput | None = None
        final_text = content.strip()
        if decision.action in ("speak", "tool") and final_text:
            output = self._to_output(final_text, trigger)
            ctx.add("assistant", output.content)
            await self._persist("assistant", output.content, trigger)

        # action=remember：写入长期记忆（不回话）
        if decision.action == "remember" and decision.memory and self.store:
            await self.store.add_long(
                decision.memory, source="self_reflection",
                importance=float(decision.importance), user=trigger.user)

        # ⑦ 会话边界 → 后台摘要 + 画像抽取
        self._maybe_summarize(ctx, trigger.user)

        return MindResult(decision=decision, output=output, agent=agent_result,
                          monologue=monologue, reflection=reflection)

    # ------------------------------------------------------------------ ② 内独白
    async def _monologue(self, desc: str, recent: str) -> str:
        try:
            resp = await self.gateway.chat([
                ChatMessage(role="system", content=f"{self.system}\n\n{prompts.MONOLOGUE_SYSTEM}"),
                ChatMessage(role="user", content=prompts.MONOLOGUE_USER.format(
                    input_desc=desc, recent=recent)),
            ], task="monologue")
        except LLMError as e:
            logger.warning("内独白调用失败：%s", e)
            return ""
        return (resp.content or "").strip()

    # ------------------------------------------------------------------ ③ 决策
    async def _decide(self, desc: str, recent: str, monologue: str,
                      memory_block: str) -> Decision:
        user_text = prompts.DECISION_USER.format(
            monologue=monologue or "（没想太多）", input_desc=desc, recent=recent)
        if memory_block:
            user_text += f"\n\n【你想起的旧事】\n{memory_block}"
        msgs = [
            ChatMessage(role="system", content=prompts.DECISION_SYSTEM),
            ChatMessage(role="user", content=user_text),
        ]
        try:
            data = await structured_chat(self.gateway, msgs, retry=self.llm_retry,
                                         task="decision")
        except StructuredError as e:
            # 兜底：不再推理循环，直接让模型用纯文本回一句（方案「降级为纯文本直发」）
            logger.warning("决策结构化失败，降级为纯文本直发：%s", e)
            return await self._degrade_plain(desc, recent, monologue)
        except LLMError as e:
            # 模型不可达（如本地 Ollama 没开）：本轮沉默，绝不让异常打断心跳循环
            logger.warning("决策调用失败，本轮沉默：%s", e)
            return Decision(action="silence", reason="LLM 调用失败")
        return self._to_decision(data)

    def _to_decision(self, data: dict[str, Any]) -> Decision:
        """把 LLM 返回的 dict 稳妥地转成 Decision（字段缺失/非法都不抛异常）。"""
        action = str(data.get("action") or "speak").strip().lower()
        if action not in _VALID_ACTIONS:
            action = "speak"

        tool: ToolCall | None = None
        if action == "tool":
            raw = data.get("tool") or {}
            if isinstance(raw, dict):
                name = str(raw.get("name") or "").strip()
                args = raw.get("args") or {}
            else:
                name, args = str(raw).strip(), {}
            if name and isinstance(args, dict):
                tool = ToolCall(name=name, args=args)
            else:
                # 说要用工具却没给名字 → 保守沉默，避免误触发
                logger.info("决策为 tool 但未给出有效工具名，转为 silence")
                action = "silence"

        try:
            importance = float(data.get("importance") or 0.5)
        except (TypeError, ValueError):
            importance = 0.5
        importance = min(max(importance, 0.0), 1.0)

        return Decision(
            action=action,                       # type: ignore[arg-type]
            content=str(data.get("content") or ""),
            tool=tool,
            memory=str(data.get("memory") or ""),
            importance=importance,
            reason=str(data.get("reason") or ""),
        )

    async def _degrade_plain(self, desc: str, recent: str, monologue: str) -> Decision:
        """结构化失败时的兜底：普通文本调用，内容直接作为回复。"""
        try:
            resp = await self.gateway.chat([
                ChatMessage(role="system", content=self.system),
                ChatMessage(role="user", content=(
                    f"【当前情况】\n{desc}\n\n【最近对话】\n{recent}\n\n"
                    "请用一句话自然地回应。")),
            ], task="decision")
        except LLMError as e:
            logger.warning("降级文本调用也失败：%s", e)
            return Decision(action="silence", reason="LLM 调用失败")
        return Decision(action="speak", content=(resp.content or "").strip(),
                        reason="结构化降级为纯文本")

    # ------------------------------------------------------------------ ⑤ 反思
    async def _reflect(self, draft: str, desc: str, decision: Decision) -> tuple[str, str]:
        """按长度/重要性触发二次修订。失败时保留原稿（反思是锦上添花，不能拖垮主链）。"""
        if not (len(draft) >= REFLECT_MIN_LEN or decision.importance >= REFLECT_IMPORTANCE):
            return draft, ""
        msgs = [
            ChatMessage(role="system", content=prompts.REFLECTION_SYSTEM),
            ChatMessage(role="user", content=(
                f"【当时的情况】\n{desc}\n\n【你本来打算这样说】\n{draft}")),
        ]
        try:
            # 反思是可选优化，不给重试次数，失败即放弃
            data = await structured_chat(self.gateway, msgs, retry=0, task="reflection")
        except (StructuredError, LLMError) as e:
            logger.info("反思未产出结果，保留原稿：%s", e)
            return draft, ""
        note = str(data.get("note") or "")
        if data.get("keep") is False and str(data.get("revised") or "").strip():
            return str(data["revised"]).strip(), note
        return draft, note

    # ------------------------------------------------------------------ 组装/落库
    def _to_output(self, content: str, trigger: CoreInput) -> CoreOutput:
        """按触发来源决定输出目标（群/私聊），继承老 CoreOutput.target 路由语义。"""
        target = "private_msg"
        group_id = ""
        if trigger.message_type == "group" and trigger.channel == "qq":
            target = "group_msg"
            group_id = trigger.group_id
        return CoreOutput(content=content, target=target,         # type: ignore[arg-type]
                          channel=trigger.channel, user=trigger.user, group_id=group_id)

    async def _persist(self, role: str, content: str, trigger: CoreInput) -> None:
        """把真实对话写入 messages 表（工作记忆的持久化备份）。"""
        if self.store and trigger.is_user_message():
            await self.store.add_message(
                role, content, user=trigger.user, channel=trigger.channel,
                user_name=trigger.user_name, ts=trigger.ts)

    def _maybe_summarize(self, ctx: SessionContext, user: str = "") -> None:
        """会话边界判定：消息数 / 空闲时长 / 跨天 → 触发后台摘要并清窗。"""
        if not (self.store and self.summarizer) or len(ctx) == 0:
            return
        max_msgs = int(self.mem_cfg.get("session_max_msgs", 30))
        idle_min = int(self.mem_cfg.get("session_idle_min", 30))
        crossed_day = (datetime.fromtimestamp(ctx.last_ts).date()
                       != datetime.now().date())
        if len(ctx) >= max_msgs or crossed_day or ctx.idle_seconds() >= idle_min * 60:
            snapshot = ctx.recent()
            self.summarizer.schedule(snapshot, user)   # 后台任务，不阻塞本轮
            ctx.clear()                                # 清窗并把计时归零，避免重复触发

    # ------------------------------------------------------------------ 工具上下文
    def _agent_context(self, ctx: SessionContext,
                       memory_block: str) -> list[ChatMessage]:
        """给 AgentLoop 的上下文：召回记忆 + 最近对话。"""
        msgs: list[ChatMessage] = []
        if memory_block:
            msgs.append(ChatMessage(role="system", content=f"【你想起的旧事】\n{memory_block}"))
        msgs.extend(ChatMessage(role=m.role, content=m.content)   # type: ignore[arg-type]
                    for m in ctx.recent(HISTORY_LIMIT))
        return msgs

    # ------------------------------------------------------------------ 输入描述
    @staticmethod
    def _describe(trigger: CoreInput) -> str:
        """把触发输入转成一句给 LLM 看的「当前情况」。"""
        if trigger.source in ("cli_message", "qq_message"):
            who = trigger.user_name or trigger.user or "对方"
            where = "（群里）" if trigger.message_type == "group" else ""
            return f"{who}{where}说：{trigger.content}"
        if trigger.source == "heartbeat":
            task = f"，任务：{trigger.task}" if trigger.task else ""
            return (f"（心跳主动唤醒{task}）现在没有新的用户消息，"
                    "你要不要主动说点什么？")
        if trigger.source == "system_event":
            return f"（系统事件 {trigger.event}：{trigger.content}）"
        return f"（{trigger.source}）{trigger.content}"