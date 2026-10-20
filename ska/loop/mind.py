"""Mind —— 自觉决策层。

一次唤醒的时序（对应方案第 2 节 ①~⑧；阶段三在②之前插入「上下文组装 + 压缩」）：
    ⓪ 权限闸（登记用户 / 拉黑拦截）
    ① 记忆召回（profile + FTS + 最近摘要，见 memory/recall.py）
    ⓵ 上下文组装（ContextBuilder → ContextPlan；接近窗口上限则压缩 + 记忆整理）
    ② 内独白（LLM，只进日志/记忆，不进用户输出）
    ③ 决策（LLM 结构化输出 → Decision：silence / speak / remember / tool）
    ④ 需要工具则交 AgentLoop 执行 think→act→observe
    ⑤ 反思（按长度/重要性触发二次修订，Reflexion）
    ⑥ 组装 MindResult（speak 走 CoreOutput 路由；remember 落库）
    ⑦ 会话边界判定 → 后台摘要 + 画像抽取
    ⑧ 返回，交给心跳循环回到 bus.wait

阶段三新增（见方案 §2.3 / §3.3 / §4.2）：
    * 多会话：trigger.session 决定窗口路由（由 BeatLoop 用 SessionRegistry 传入 ctx）；
    * 沙盒会话（trigger.sandbox）：照常推理，但不写 messages / long / 画像 / 摘要；
    * 上下文工程：ContextPlan 决定「要多少上下文」，超 compress_at 触发压缩；
    * MindLine 插桩：每一步 emit 一条 TraceEvent（lane ∈ input/think/act）。
    * 工具感知：决策前把「可用工具清单」注入提示词，否则模型压根不知道能调工具；
    * 自主路由：决策可携带 route，让 Agent 自己决定这条输出投到哪个渠道/会话；
    * 自主休眠：决策可携带 sleep_seconds，交给 BeatLoop 覆盖默认心跳节奏。

关键点：老项目 prompt.json 的 self_response「隐式自循环」在这里被显式拆成 ②③④，
不再有「把自己的输出再塞回输入」的递归风险。
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from ..admin.trace import emit_trace
from ..kernel.context import Message, SessionContext
from ..kernel.packet import (CoreInput, CoreOutput, Decision, MindResult, Route,
                             ToolCall)
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
# 压缩时至少保留的最近消息条数（避免压缩后对话体验断层）
COMPRESS_KEEP_MIN = 2
# Agent 自主休眠上限（秒）：防止模型给出离谱数值把心跳彻底停死（24 小时兜底）
MAX_SLEEP_SECONDS = 86400.0

_VALID_ACTIONS = {"speak", "silence", "remember", "tool"}


def format_history(msgs: list[Message]) -> str:
    """把工作记忆格式化成人类可读的对话文本（阶段二提示词结构，ContextPlan 同款）。"""
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
        bus: Any = None,
        builder: Any = None,
    ) -> None:
        self.gateway = gateway
        self.agent = agent            # AgentLoop
        self.store = store            # MemoryStore（可为 None：无记忆也能跑）
        self.summarizer = summarizer  # Summarizer（可为 None）
        self.cfg = cfg
        self.policy = policy          # admin.policy.Policy（阶段二；可为 None=不裁权限）
        self.persona_prompt = persona_prompt      # 保留原始人设，供后台切档时重建
        self.system = prompts.make_system(persona_prompt)
        self.bus = bus                # 事件总线（MindLine 插桩用；可为 None）
        self.builder = builder        # ContextBuilder（阶段三；可为 None=退回阶段二行为）
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
        session = str(getattr(trigger, "session", "") or "default")
        sandbox = bool(getattr(trigger, "sandbox", False))
        trace_id = uuid.uuid4().hex[:12]
        # 三条插桩要共用的上下文（用 dict 展开，避免每处重复五个参数）
        tb: dict[str, Any] = {"trace_id": trace_id, "session": session, "sandbox": sandbox}

        # ⓪ 权限闸（阶段二）：登记聊天用户；被拉黑者直接忽略（不回话、不进记忆）
        if self.policy is not None and trigger.is_user_message():
            await self.policy.note_user(trigger)
            if not await self.policy.should_respond(trigger):
                await emit_trace(self.bus, lane="input", kind="silence",
                                 title="权限闸拒绝", detail=trigger.content, **tb)
                return MindResult(decision=Decision(action="silence", reason="用户不可回应"))

        # 输入轨迹（lane=input）
        if trigger.is_user_message():
            await emit_trace(self.bus, lane="input", kind="user_msg",
                             title=f"{trigger.user_name or trigger.user}：{trigger.content[:40]}",
                             detail=trigger.content, **tb)
        else:
            await emit_trace(self.bus, lane="input", kind="wake",
                             title=f"唤醒（{trigger.source}）", detail=trigger.content, **tb)

        # ① 记忆召回
        memory_block = ""
        if self.store:
            memory_block = await recall.build_memory_block(
                self.store, trigger.content, user=trigger.user, cfg=self.mem_cfg)

        # 记录用户输入到工作记忆（+ 原始流水落库；沙盒会话不落库）
        if trigger.is_user_message():
            ctx.add("user", f"{trigger.user_name or trigger.user}: {trigger.content}")
            await self._persist("user", trigger.content, trigger)

        # ⓵ 上下文组装 + 需要时压缩（阶段三 §3.3）
        plan = self._build_plan(ctx, memory_block)
        plan = await self._maybe_compress(ctx, plan, memory_block=memory_block,
                                          user=trigger.user, **tb)
        recent = (plan.render_history() if plan is not None
                  else format_history(ctx.recent(HISTORY_LIMIT)))
        tokens = plan.used_tokens if plan is not None else 0
        desc = self._describe(trigger)

        # 可用工具集（阶段三增强）：必须在决策之前算好——决策 LLM 得先「知道自己有什么工具」，
        # 否则它永远只会选 speak / silence，压根想不到 action=tool（这是「工具感知」的关键）。
        allow = await self.policy.tools_for(trigger) if self.policy is not None else None
        catalog = self._tool_catalog(allow)

        # ② 内独白
        monologue = await self._monologue(desc, recent)
        if monologue:
            await emit_trace(self.bus, lane="think", kind="monologue",
                             title="内心独白", detail=monologue, **tb)

        # ③ 决策
        decision, degrade = await self._decide_with_trace(desc, recent, monologue,
                                                          memory_block, tokens, tb, catalog)

        # ④ 工具执行（allow 已在上方算好；非用户消息路径为 None=不裁剪）
        agent_result = None
        content = decision.content
        if decision.action == "tool":
            agent_result = await self.agent.run(
                self._agent_context(ctx, memory_block, plan), first=decision.tool, allow=allow,
                trace_id=trace_id, session=session, sandbox=sandbox)
            content = agent_result.final

        # ⑤ 反思（仅对真实用户消息做质量把关，省下主动路径的调用）
        reflection = ""
        if trigger.is_user_message() and content.strip():
            content, reflection = await self._reflect(content, desc, decision)
            if reflection:
                await emit_trace(self.bus, lane="think", kind="reflection",
                                 title="反思", detail=reflection, **tb)

        # ⑥ 组装输出
        output: CoreOutput | None = None
        final_text = content.strip()
        if decision.action in ("speak", "tool") and final_text:
            output = self._to_output(final_text, trigger, decision)
            # 多会话：让 webchat 渠道能投回本会话。若 Agent 已用 route 显式指定目标会话，
            # 则尊重它的选择，不再用触发会话覆盖（阶段三增强：自主路由）。
            if not (decision.route and decision.route.session):
                output.session = session
            ctx.add("assistant", output.content)
            await self._persist("assistant", output.content, trigger)
            await emit_trace(self.bus, lane="act", kind="output",
                             title=f"输出 → {output.channel}", detail=output.content,
                             tokens=tokens, **tb)

        # action=remember：写入长期记忆（不回话；沙盒会话不写）
        if (decision.action == "remember" and decision.memory and self.store
                and not sandbox):
            await self.store.add_long(
                decision.memory, source="self_reflection",
                importance=float(decision.importance), user=trigger.user)
            await emit_trace(self.bus, lane="think", kind="memory_write",
                             title="写入长期记忆", detail=decision.memory, **tb)

        # ⑦ 会话边界 → 后台摘要 + 画像抽取（沙盒会话跳过）
        self._maybe_summarize(ctx, trigger.user, sandbox=sandbox)

        return MindResult(decision=decision, output=output, agent=agent_result,
                          monologue=monologue, reflection=reflection)

    # ------------------------------------------------------------------ 上下文组装（阶段三）
    def _build_plan(self, ctx: SessionContext, memory_block: str) -> Any:
        """用 ContextBuilder 组装一次上下文计划；无 builder 时返回 None（退回阶段二）。"""
        if self.builder is None:
            return None
        try:
            return self.builder.build(system=self.system, memory_block=memory_block,
                                      history=ctx.recent(HISTORY_LIMIT))
        except Exception:  # noqa: BLE001 —— 组装失败不该打断对话
            logger.exception("上下文组装失败，退回阶段二行为")
            return None

    async def _maybe_compress(self, ctx: SessionContext, plan: Any, *,
                              memory_block: str, user: str, session: str,
                              sandbox: bool, trace_id: str) -> Any:
        """占用率达到 compress_at → 滚动摘要 + 记忆整理 + 重建上下文。

        沙盒会话：只做「内存侧压缩」（摘要文本不落库），窗口照样收窄，
        以保证长会话不撑爆模型窗口，同时不污染任何记忆库。
        """
        if plan is None or not plan.need_compress or self.summarizer is None:
            return plan
        msgs = ctx.snapshot()
        if len(msgs) <= COMPRESS_KEEP_MIN:
            return plan
        # 保留最近的若干条，把更早的一段交给摘要吸收
        keep = max(len(msgs) // 2, COMPRESS_KEEP_MIN)
        old = msgs[:-keep]
        if not old:
            return plan
        try:
            summary = await self.summarizer.organize(old, user, write=not sandbox)
        except Exception:  # noqa: BLE001 —— 压缩是优化，失败就沿用原窗口
            logger.exception("上下文压缩失败，沿用原窗口")
            return plan
        if not summary:
            return plan
        ctx.drop_oldest(len(old))                     # 截掉已被摘要吸收的旧段
        ctx.add("system", f"【早前对话摘要】{summary}")   # 摘要留在窗口，供后续轮次可见
        await emit_trace(self.bus, lane="think", kind="compress",
                         title=f"压缩上下文：摘要 {len(old)} 条旧消息",
                         detail=summary, tokens=plan.used_tokens,
                         trace_id=trace_id, session=session, sandbox=sandbox)
        if not sandbox:
            await emit_trace(self.bus, lane="think", kind="memory_write",
                             title="记忆整理：写入长期记忆与画像", detail=summary,
                             trace_id=trace_id, session=session, sandbox=sandbox)
        return self._build_plan(ctx, memory_block)     # 重建上下文

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
    async def _decide_with_trace(self, desc: str, recent: str, monologue: str,
                                 memory_block: str, tokens: int,
                                 tb: dict[str, Any], catalog: str = "") -> tuple[Decision, bool]:
        """执行决策并 emit 轨迹；返回 (decision, 是否发生了降级)。

        catalog：本轮可用工具清单（名字 + 说明）。注入决策提示词，让模型「知道自己有什么工具」。
        """
        tools_block = (f"\n\n【你可用的工具（action=tool 时从中选）】\n{catalog}"
                       if catalog else "\n\n【你可用的工具】\n（本轮没有可用工具）")
        user_text = prompts.DECISION_USER.format(
            monologue=monologue or "（没想太多）", input_desc=desc, recent=recent,
            tools=tools_block)
        if memory_block:
            user_text += f"\n\n【你想起的旧事】\n{memory_block}"
        msgs = [
            ChatMessage(role="system", content=prompts.DECISION_SYSTEM),
            ChatMessage(role="user", content=user_text),
        ]
        degrade = False
        try:
            data = await structured_chat(self.gateway, msgs, retry=self.llm_retry,
                                         task="decision")
            decision = self._to_decision(data)
        except StructuredError as e:
            # 兜底：不再推理循环，直接让模型用纯文本回一句（方案「降级为纯文本直发」）
            logger.warning("决策结构化失败，降级为纯文本直发：%s", e)
            degrade = True
            decision = await self._degrade_plain(desc, recent, monologue)
        except LLMError as e:
            # 模型不可达（如本地 Ollama 没开）：本轮沉默，绝不让异常打断心跳循环
            logger.warning("决策调用失败，本轮沉默：%s", e)
            await emit_trace(self.bus, lane="think", kind="error",
                             title="决策调用失败", detail=str(e), **tb)
            return Decision(action="silence", reason="LLM 调用失败"), False

        await emit_trace(self.bus,
                         lane="think", kind="degrade" if degrade else "decision",
                         title=f"决策：{decision.action}（{decision.reason or '无理由'}）",
                         detail=decision.content or decision.memory or decision.reason,
                         tokens=tokens, **tb)
        return decision, degrade

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
            route=self._to_route(data.get("route")),
            sleep_seconds=self._to_sleep(data.get("sleep_seconds")),
        )

    @staticmethod
    def _to_route(raw: Any) -> Route | None:
        """把决策 JSON 里的 route 稳妥地转成 Route（非法/全空 → None=沿用触发来路）。"""
        if not isinstance(raw, dict):
            return None
        fields = {k: str(raw.get(k) or "").strip() for k in
                  ("channel", "user", "session", "message_type", "group_id")}
        if not any(fields.values()):            # 全空等价于「没指定」，避免无意义的覆盖
            return None
        return Route(**fields)

    @staticmethod
    def _to_sleep(raw: Any) -> float:
        """把决策 JSON 里的 sleep_seconds 转成受限的秒数（0=按默认心跳节奏）。"""
        try:
            seconds = float(raw or 0)
        except (TypeError, ValueError):
            return 0.0
        return min(max(seconds, 0.0), MAX_SLEEP_SECONDS)

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
    def _to_output(self, content: str, trigger: CoreInput,
                   decision: Decision | None = None) -> CoreOutput:
        """决定输出目标：优先用 Agent 自主指定的 route，缺省沿用触发来路（兼容旧行为）。

        路由优先级：decision.route（逐字段覆盖） > 触发来路（trigger）。
        message_type=group 且渠道为 qq 时投递到群，否则按私聊投递。
        """
        channel = trigger.channel
        user = trigger.user
        session = str(getattr(trigger, "session", "") or "")
        message_type = trigger.message_type
        group_id = trigger.group_id

        route = getattr(decision, "route", None) if decision is not None else None
        if route is not None:                      # 逐字段覆盖：空字段=沿用触发来路
            channel = route.channel or channel
            user = route.user or user
            session = route.session or session
            message_type = route.message_type or message_type
            group_id = route.group_id or group_id

        target = "private_msg"
        if message_type == "group" and channel == "qq":
            target = "group_msg"
        return CoreOutput(content=content, target=target,         # type: ignore[arg-type]
                          channel=channel, user=user, group_id=group_id, session=session)

    async def _persist(self, role: str, content: str, trigger: CoreInput) -> None:
        """把真实对话写入 messages 表（工作记忆的持久化备份）。

        沙盒会话（trigger.sandbox）跳过：照常推理，但不写任何记忆库（方案 §2.3）。
        """
        if trigger.sandbox:
            return
        if self.store and trigger.is_user_message():
            await self.store.add_message(
                role, content, user=trigger.user, channel=trigger.channel,
                user_name=trigger.user_name, ts=trigger.ts)

    def _maybe_summarize(self, ctx: SessionContext, user: str = "",
                         *, sandbox: bool = False) -> None:
        """会话边界判定：消息数 / 空闲时长 / 跨天 → 触发后台摘要并清窗。

        沙盒会话不触发（不写 sessions / long / 画像）。
        """
        if sandbox or not (self.store and self.summarizer) or len(ctx) == 0:
            return
        max_msgs = int(self.mem_cfg.get("session_max_msgs", 30))
        idle_min = int(self.mem_cfg.get("session_idle_min", 30))
        crossed_day = (datetime.fromtimestamp(ctx.last_ts).date()
                       != datetime.now().date())
        if len(ctx) >= max_msgs or crossed_day or ctx.idle_seconds() >= idle_min * 60:
            snapshot = ctx.recent()
            self.summarizer.schedule(snapshot, user)   # 后台任务，不阻塞本轮
            ctx.clear()                                # 清窗并把计时归零，避免重复触发

    # ------------------------------------------------------------------ 工具感知
    def _tool_catalog(self, allow: set[str] | None) -> str:
        """列出本轮可用工具（名字 + 说明），供决策 LLM 知道自己能调用什么。

        为什么必须有这一步：AgentLoop 虽会把工具声明（schema）传给 LLM，但那发生在
        action=tool 之后；若**决策阶段**看不到工具清单，模型根本不会想到用工具，
        于是永远只输出 speak/silence —— 表现就是「智能体不知道自己是能调工具的」。

        allow 为 None 表示不裁剪（非用户消息路径）；为集合时只列集合内的工具。
        返回空串表示本轮无可用工具（决策提示词会据此提示「只能用 speak/silence/remember」）。
        """
        tools = getattr(self.agent, "tools", {}) or {}
        names = sorted(tools) if allow is None else sorted(n for n in tools if n in allow)
        lines = []
        for name in names:
            desc = str(getattr(tools[name], "description", "") or "").strip().splitlines()
            lines.append(f"- {name}：{desc[0] if desc else '（无说明）'}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ 工具上下文
    def _agent_context(self, ctx: SessionContext, memory_block: str,
                       plan: Any = None) -> list[ChatMessage]:
        """给 AgentLoop 的上下文：优先用 ContextPlan（含记忆块 + 预算内历史）。"""
        if plan is not None and plan.messages:
            return [ChatMessage(role=m["role"], content=m["content"])   # type: ignore[arg-type]
                    for m in plan.messages]
        # 退路（无 builder）：召回记忆 + 最近对话（阶段二行为）
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
        if trigger.source in ("cli_message", "qq_message", "webchat_message"):
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
