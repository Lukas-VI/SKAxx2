"""AgentLoop —— 执行者：think → act → observe 工具链。

职责：一旦 Mind 决定「需要做点什么」，由本类负责把它一步步做完（调工具、拿结果、成文）。
每轮最多调 1 个工具，总步数上限 max_steps（默认 5），到顶就截断（stopped=True），
避免老项目那种「模型反复自循环、步数失控」的问题。

与 LLM 的交互刻意采用「观察结果作为 system 消息回填」的方式，
而不是严格的 assistant.tool_calls / tool_call_id 协议往返：
Ollama 与 DeepSeek 都能稳定接受后者，换来的是更少的协议耦合与更简单的实现。
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..admin.trace import emit_trace
from ..kernel.packet import AgentResult, AgentStep, ToolCall
from ..llm.gateway import LLMError
from ..llm.models import ChatMessage
from ..tools.base import Tool, tool_schemas

logger = logging.getLogger(__name__)

MAX_STEPS = 5

# 工具结果截断上限（字符）：本地模型上下文有限，且工具可能吐回超长文本，
# 不截断会把上下文撑爆（方案 §3.2 第 5 层「工具结果按需截断」）。
MAX_OBS_CHARS = 2000

# 同一调用「连续重复」多少次即判定为死循环、提前终止。
# 只是 max_steps 之外的「更早止损」：模型偶尔重查一次是正常的，
# 但连续 3 次一字不差地调同一个工具同一组参数，几乎必然是卡住了。
MAX_REPEATS = 2


class AgentLoop:
    """工具链执行者。"""

    def __init__(
        self,
        gateway: Any,
        tools: dict[str, Tool],
        system_prompt: str,
        max_steps: int = MAX_STEPS,
        *,
        bus: Any = None,
        max_obs_chars: int = MAX_OBS_CHARS,
    ) -> None:
        self.gateway = gateway
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.bus = bus                          # 事件总线（MindLine 插桩用；可为 None）
        self.max_obs_chars = max(int(max_obs_chars), 100)

    async def run(
        self,
        context: list[ChatMessage],
        *,
        first: ToolCall | None = None,
        allow: set[str] | None = None,
        trace_id: str = "",
        session: str = "",
        sandbox: bool = False,
    ) -> AgentResult:
        """执行工具链。

        Args:
            context: 起始对话上下文（不含 system，本方法会自己加上执行者 system）。
            first:   若 Mind 已明确指定要先调用的工具，从这里直接开跑。
            allow:   允许使用的工具名集合（阶段二权限裁剪）；None 表示不裁剪。
            trace_id/session/sandbox: 阶段三 MindLine 插桩上下文（透传给 _exec）。

        Returns:
            AgentResult：final 为最终文本，steps 为轨迹，stopped 表示是否因超步截断。
        """
        convo: list[ChatMessage] = [ChatMessage(role="system", content=self.system_prompt)]
        convo.extend(context)
        # 只在「允许集合」内声明工具；空集会让 tools 参数为 None，模型便不再要工具
        schemas = tool_schemas(self.tools, allow) or None
        steps: list[AgentStep] = []
        step_no = 0
        trace_kw = {"trace_id": trace_id, "session": session, "sandbox": sandbox}
        last_sig: str | None = None      # 上一次工具调用的「签名」（名字 + 参数），用于查重
        repeats = 0                      # 该签名已连续重复的次数

        # 若决策已点名工具，先执行它（记为第 1 步，不额外消耗一次 LLM 调用）
        if first is not None:
            step_no = 1
            obs = await self._exec(first, allow, **trace_kw)
            steps.append(AgentStep(step=1, thought="（按决策直接调用工具）",
                                   tool=first.name, args=first.args, observation=obs))
            convo.append(self._observation_msg(first.name, obs))
            last_sig = self._signature(first)

        while step_no < self.max_steps:
            step_no += 1
            try:
                resp = await self.gateway.chat(convo, tools=schemas, task="agent")
            except LLMError as e:
                logger.warning("AgentLoop LLM 调用失败：%s", e)
                return AgentResult(final="", steps=steps, error=str(e))

            if resp.tool_call is not None:
                sig = self._signature(resp.tool_call)
                repeats = repeats + 1 if sig == last_sig else 0
                last_sig = sig
                if repeats >= MAX_REPEATS:
                    # 连续重复同一调用：多半是模型卡住了，提前止损（比耗尽 max_steps 更好）
                    logger.info("AgentLoop 检测到重复调用 %s，提前终止", resp.tool_call.name)
                    await emit_trace(self.bus, lane="act", kind="error",
                                     title=f"重复调用被终止：{resp.tool_call.name}",
                                     detail=f"参数：{resp.tool_call.args}", **trace_kw)
                    steps.append(AgentStep(step=step_no, thought=resp.content,
                                           tool=resp.tool_call.name, args=resp.tool_call.args,
                                           observation="（检测到重复调用，已终止）"))
                    return AgentResult(final="", steps=steps, stopped=True)
                obs = await self._exec(resp.tool_call, allow, **trace_kw)
                steps.append(AgentStep(step=step_no, thought=resp.content,
                                       tool=resp.tool_call.name, args=resp.tool_call.args,
                                       observation=obs))
                convo.append(ChatMessage(role="assistant", content=resp.content))
                convo.append(self._observation_msg(resp.tool_call.name, obs))
                continue

            # 模型不再要求调工具 → 视为最终回答
            steps.append(AgentStep(step=step_no, thought=resp.content))
            return AgentResult(final=(resp.content or "").strip(), steps=steps)

        # 步数用尽
        logger.info("AgentLoop 达到最大步数 %d，截断", self.max_steps)
        return AgentResult(final="", steps=steps, stopped=True)

    # ------------------------------------------------------------------ 内部
    async def _exec(self, call: ToolCall, allow: set[str] | None = None, *,
                    trace_id: str = "", session: str = "", sandbox: bool = False) -> str:
        """执行一次工具调用，把任何异常都转成可读文本（不让单个工具崩掉整轮）。

        allow 非 None 时做二次权限校验：即使模型（可能被诱导）点名了越权工具，
        这里也不执行——权限只在装配层判定是不够的，执行前必须再确认一次。

        阶段三：前后各发一条 MindLine 轨迹（tool_call / tool_result），
        并在返回前对结果做长度截断（方案 §3.2 第 5 层）。
        """
        if allow is not None and call.name not in allow:
            logger.info("工具 %s 不在当前用户可用集合内，拒绝执行", call.name)
            await emit_trace(self.bus, trace_id=trace_id, session=session, lane="act",
                             kind="error", title=f"工具越权被拒：{call.name}",
                             detail=f"可用工具：{sorted(allow)}", sandbox=sandbox)
            return f"你没有权限使用工具 {call.name}。可用工具：{sorted(allow)}"
        tool = self.tools.get(call.name)
        if tool is None:
            return f"没有名为 {call.name} 的工具。可用工具：{sorted(self.tools)}"

        await emit_trace(self.bus, trace_id=trace_id, session=session, lane="act",
                         kind="tool_call", title=f"调用工具：{call.name}",
                         detail=f"参数：{call.args}", sandbox=sandbox)
        started = time.time()
        try:
            result = await tool.run(**call.args)
        except Exception as e:  # noqa: BLE001
            logger.exception("工具 %s 执行异常", call.name)
            obs = f"工具执行失败：{e}"
            await emit_trace(self.bus, trace_id=trace_id, session=session, lane="act",
                             kind="error", title=f"工具异常：{call.name}",
                             detail=str(e), sandbox=sandbox)
            return obs
        obs = str(result)
        elapsed = time.time() - started
        truncated, _ = self._clip(obs)
        await emit_trace(self.bus, trace_id=trace_id, session=session, lane="act",
                         kind="tool_result",
                         title=f"工具返回：{call.name}（{elapsed:.2f}s）",
                         detail=truncated, sandbox=sandbox)
        return truncated

    def _clip(self, text: str) -> tuple[str, bool]:
        """按字符上限截断工具输出（返回 截断后文本, 是否发生截断）。"""
        if len(text) <= self.max_obs_chars:
            return text, False
        return text[:self.max_obs_chars] + f"\n…（已截断，原始 {len(text)} 字符）", True

    @staticmethod
    def _signature(call: ToolCall) -> str:
        """给一次工具调用生成稳定签名（名字 + 排序后的参数），用于「重复调用」查重。

        用 json.dumps(sort_keys=True) 而不是直接 hash(dict)：参数字典顺序不影响签名，
        且各种不可序列化值（如自定义对象）用 default=str 兜底，保证不抛异常。
        """
        return f"{call.name}:{json.dumps(call.args, sort_keys=True, default=str, ensure_ascii=False)}"

    @staticmethod
    def _observation_msg(name: str, obs: str) -> ChatMessage:
        """把工具观察结果包装成 system 消息回填上下文。"""
        return ChatMessage(role="system", content=f"[工具 {name} 的结果]\n{obs}")