"""loop 层测试：FakeLLM（网关鸭子类型替身）驱动 Mind / AgentLoop。

覆盖：
    * AgentLoop：think→act→observe 链路、超步截断、工具异常兜底；
    * Mind：speak / silence / remember / tool 决策与 MindResult 组装；
    * 结构化失败时的降级直发。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any

from ska.kernel.context import SessionContext
from ska.kernel.packet import CoreInput, ToolCall
from ska.llm.models import ChatMessage, LLMResponse
from ska.loop.agent import AgentLoop
from ska.loop.mind import Mind
from ska.memory.store import MemoryStore


# --------------------------------------------------------------------------- FakeLLM
class FakeLLM:
    """可编程的 LLM 替身：按顺序弹脚本，弹完重复最后一条。

    script 元素：(content, tool_call|None)
    """

    def __init__(self, script: list[tuple[str, ToolCall | None]] | None = None) -> None:
        self.script = script or []
        self.i = 0
        self.calls: list[list[ChatMessage]] = []   # 记录每次调用，供断言

    def _next(self) -> tuple[str, ToolCall | None]:
        idx = min(self.i, len(self.script) - 1)
        self.i += 1
        return self.script[idx]

    async def chat(self, messages, *, tools=None, json_mode=False,
                   temperature=None, max_tokens=None, task="") -> LLMResponse:
        self.calls.append(list(messages))
        content, tool_call = self._next()
        return LLMResponse(content=content, tool_call=tool_call, model="fake")


class BrokenToolLLM(FakeLLM):
    """总是要求调用一个会导致执行异常的工具。"""

    async def chat(self, messages, *, tools=None, json_mode=False,
                   temperature=None, max_tokens=None, task="") -> LLMResponse:
        self.calls.append(list(messages))
        return LLMResponse(content="我需要工具", model="fake",
                           tool_call=ToolCall(name="boom_tool", args={}))


class _BoomTool:
    name = "boom_tool"

    async def run(self, **kwargs: Any) -> str:
        raise RuntimeError("炸弹工具炸了")

    def schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name,
                                                 "description": "", "parameters": {}}}


# --------------------------------------------------------------------------- AgentLoop
class TestAgentLoop(unittest.IsolatedAsyncioTestCase):
    def _tools(self) -> dict[str, Any]:
        class FakeTimeTool:
            name = "time"

            async def run(self, **kwargs: Any) -> str:
                return "2026-10-12 12:00:00"

            def schema(self) -> dict[str, Any]:
                return {"type": "function",
                        "function": {"name": "time", "description": "",
                                     "parameters": {"type": "object",
                                                    "properties": {}}}}

        return {"time": FakeTimeTool(), "boom_tool": _BoomTool()}

    async def test_think_act_observe(self):
        """模型先要求调 time 工具，观察后再给出最终回答。"""
        llm = FakeLLM([
            ("我需要查一下时间", ToolCall(name="time", args={})),
            ("现在是2026-10-12 12:00:00，记得按时吃饭。", None),
        ])
        loop = AgentLoop(llm, self._tools(), "你是助手")
        ctx = [ChatMessage(role="user", content="现在几点？")]
        result = await loop.run(ctx)

        self.assertIn("2026-10-12", result.final)
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.steps[0].tool, "time")
        self.assertFalse(result.stopped)

    async def test_direct_final_answer(self):
        """模型不需要工具时直接回答。"""
        llm = FakeLLM([("嗨，你好呀。", None)])
        loop = AgentLoop(llm, self._tools(), "你是助手")
        result = await loop.run([ChatMessage(role="user", content="在吗")])
        self.assertEqual(result.final, "嗨，你好呀。")
        self.assertEqual(len(result.steps), 1)

    async def test_max_steps_truncation(self):
        """模型不停要工具 → 步数用尽被截断，不会无限循环。"""
        llm = FakeLLM([("继续查", ToolCall(name="time", args={})), ])
        loop = AgentLoop(llm, self._tools(), "你是助手", max_steps=3)
        result = await loop.run([ChatMessage(role="user", content="查吧")])
        self.assertTrue(result.stopped)
        self.assertLessEqual(len(result.steps), 3)

    async def test_tool_error_does_not_crash_loop(self):
        """工具抛异常 → 观察文本转成「工具执行失败」，循环继续。"""
        llm = BrokenToolLLM()
        loop = AgentLoop(llm, self._tools(), "你是助手", max_steps=2)
        result = await loop.run([ChatMessage(role="user", content="boom")])
        # 第二拍模型仍要求工具 → 步数用尽截断；关键是没有异常抛出
        self.assertTrue(result.stopped or result.final == "")


# --------------------------------------------------------------------------- Mind
class TestMind(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(os.path.join(self._tmp.name, "mind.db"))

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self._tmp.cleanup()

    def _mind(self, llm: FakeLLM, **kw: Any) -> Mind:
        loop = AgentLoop(llm, {}, "你是助手")
        return Mind(llm, loop, store=kw.pop("store", self.store), cfg=kw.pop("cfg", None),
                    persona_prompt="你叫赛里斯库妮娅，是个有温度的人。")

    def _input(self, text: str = "你好") -> CoreInput:
        return CoreInput(source="cli_message", content=text, user="u",
                         user_name="小明", channel="cli")

    async def test_speak_decision(self):
        """决策 speak → MindResult 带可见输出，且工作记忆记录了对话。"""
        llm = FakeLLM([
            ("他在跟我打招呼，心情不错。", None),          # 内独白
            ('{"action":"speak","content":"你好呀，小明！","importance":0.4,"reason":"礼貌回应"}', None),
        ])
        mind = self._mind(llm)
        ctx = SessionContext(20)
        result = await mind.wake(ctx, self._input())

        self.assertEqual(result.decision.action, "speak")
        self.assertEqual(result.output.content, "你好呀，小明！")
        self.assertTrue(result.output.is_visible())
        # 记忆落库 + 工作记忆
        self.assertIn("小明", ctx.to_chat()[-2]["content"])
        rows = await self.store.recent_messages(10, "u")
        self.assertEqual(len(rows), 2)   # user + assistant

    async def test_silence_decision(self):
        """决策 silence → 无输出。"""
        llm = FakeLLM([
            ("他只是在测试我。", None),
            ('{"action":"silence","content":"","importance":0.1,"reason":"无需回应"}', None),
        ])
        result = await self._mind(llm).wake(SessionContext(20), self._input())
        self.assertEqual(result.decision.action, "silence")
        self.assertIsNone(result.output)

    async def test_remember_decision_writes_long_memory(self):
        """决策 remember → 写入长期记忆，不回话。"""
        llm = FakeLLM([
            ("他告诉我一件重要的事。", None),
            ('{"action":"remember","memory":"小明怕打针","importance":0.9,"reason":"重要事实"}', None),
        ])
        result = await self._mind(llm).wake(SessionContext(20), self._input())
        self.assertEqual(result.decision.action, "remember")
        self.assertIsNone(result.output)
        hits = await self.store.search_long("小明怕打针", user="u")
        self.assertTrue(any("小明怕打针" in h for h in hits))

    async def test_tool_decision_runs_agent(self):
        """决策 tool → 交给 AgentLoop 执行，输出用执行结果。"""
        llm = FakeLLM([
            ("他问时间，我需要工具。", None),
            ('{"action":"tool","tool":{"name":"fake_time","args":{}},"content":"","importance":0.3,"reason":"查时间"}', None),
            ("现在是中午十二点整。", None),
        ])
        class FakeTimeTool:
            name = "fake_time"

            async def run(self, **kwargs: Any) -> str:
                return "12:00"

            def schema(self) -> dict[str, Any]:
                return {"type": "function",
                        "function": {"name": "fake_time", "description": "",
                                     "parameters": {"type": "object", "properties": {}}}}

        loop = AgentLoop(llm, {"fake_time": FakeTimeTool()}, "你是助手")
        mind = Mind(llm, loop, store=self.store, persona_prompt="你叫小赛。")
        result = await mind.wake(SessionContext(20), self._input("现在几点了"))

        self.assertEqual(result.decision.action, "tool")
        self.assertIsNotNone(result.agent)
        self.assertEqual(result.output.content, "现在是中午十二点整。")

    async def test_structured_failure_degrades_to_plain(self):
        """决策 JSON 解析失败 → 降级为纯文本直发（不回退到再推理死循环）。"""
        llm = FakeLLM([
            ("他在问我在不在。", None),
            ("这不是JSON", None),          # 结构化失败
            ("我在的，有什么需要帮忙的？", None),   # 降级文本
        ])
        result = await self._mind(llm).wake(SessionContext(20), self._input())
        self.assertEqual(result.decision.action, "speak")
        self.assertIn("我在", result.output.content)

    async def test_memory_recall_included_in_decision_prompt(self):
        """有长期记忆时，决策提示词应包含召回内容（构建记忆块）。

        用方案验收里的原句做用例：记忆「我喜欢酒心巧克力」+ 提问「我之前说过我喜欢吃什么」。
        """
        await self.store.add_long("我喜欢酒心巧克力，讨厌打针", importance=0.9, user="u")
        llm = FakeLLM([
            ("想起他喜欢巧克力。", None),
            ('{"action":"speak","content":"对呀，我记得你喜欢酒心巧克力。","importance":0.5,"reason":"基于记忆"}', None),
        ])
        result = await self._mind(llm).wake(SessionContext(20), self._input("我之前说过我喜欢吃什么"))
        # 决策用户消息应带上「你想起的旧事」
        decision_call = llm.calls[1]
        user_text = decision_call[-1].content
        self.assertIn("酒心巧克力", user_text)
        self.assertIn("酒心巧克力", result.output.content)


if __name__ == "__main__":
    unittest.main()