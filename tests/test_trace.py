"""MindLine 轨迹测试（阶段三 · 方案 §4 / 决策 E4）。

覆盖三块：
    * emit_trace —— 无总线时静默、有总线时广播；
    * TraceLog —— 分配全局递增 seq、内存环形缓冲、落库到 traces 表、清空；
    * Mind 插桩 —— 一次唤醒应产生 input / think / act 三行轨迹，且归属同一会话。

「trace 是 log 的一部分」（E4）：TraceLog 写入时既落库，也走 ska.trace 日志。
"""
from __future__ import annotations

import os
import tempfile
import unittest
from typing import Any

from ska.admin.store import AdminStore
from ska.admin.trace import TraceLog, emit_trace
from ska.context import ContextBuilder
from ska.kernel.bus import EventBus
from ska.kernel.context import SessionContext
from ska.kernel.event import TOPIC_TRACE, Event
from ska.kernel.packet import CoreInput, TraceEvent
from ska.llm.models import LLMResponse
from ska.loop.agent import AgentLoop
from ska.loop.mind import Mind
from ska.memory.store import MemoryStore


# =========================================================================== TraceLog
class TestTraceLog(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = AdminStore(os.path.join(self._tmp.name, "admin.db"))
        self.bus = EventBus()
        self.log = TraceLog(self.store, buffer=10)
        self.log.attach(self.bus)

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self._tmp.cleanup()

    async def test_emit_assigns_increasing_seq(self):
        await emit_trace(self.bus, trace_id="t1", session="s1", lane="input",
                         kind="user_msg", title="收到", detail="你好")
        await emit_trace(self.bus, trace_id="t1", session="s1", lane="think",
                         kind="decision", title="决定回话", tokens=42)
        items = self.log.recent()
        self.assertEqual([x["seq"] for x in items], [1, 2])          # 全局递增
        self.assertEqual(items[1]["tokens"], 42)
        self.assertEqual(items[1]["lane"], "think")
        self.assertEqual(self.log.last_seq(), 2)

    async def test_recent_incremental_and_session_filter(self):
        await emit_trace(self.bus, trace_id="a", session="s1", lane="think", kind="decision")
        await emit_trace(self.bus, trace_id="b", session="s2", lane="think", kind="decision")
        self.assertEqual([x["seq"] for x in self.log.recent(after=1)], [2])
        self.assertEqual([x["session"] for x in self.log.recent(session="s2")], ["s2"])

    async def test_persisted_to_store(self):
        await emit_trace(self.bus, trace_id="t1", session="s1", lane="act",
                         kind="tool_call", title="调用 time")
        rows = await self.store.list_traces(after=0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["kind"], "tool_call")
        self.assertEqual(rows[0]["title"], "调用 time")
        self.assertEqual((await self.store.stats())["traces"], 1)

    async def test_emit_without_bus_is_noop(self):
        self.assertIsNone(await emit_trace(None, trace_id="x", session="",
                                           lane="think", kind="decision"))

    async def test_invalid_kind_is_skipped_not_raised(self):
        """轨迹是旁路观测：kind 拼错时只丢这一条，绝不能让整轮对话抛异常。"""
        self.assertIsNone(await emit_trace(self.bus, trace_id="t1", session="s1",
                                           lane="input", kind="not_a_kind"))
        self.assertEqual(self.log.recent(), [])

    async def test_image_and_sticker_kinds_accepted(self):
        """阶段四新增的两种轨迹类型必须在白名单里。"""
        await emit_trace(self.bus, trace_id="t1", session="s1", lane="input",
                         kind="image", title="图片解析")
        await emit_trace(self.bus, trace_id="t1", session="s1", lane="act",
                         kind="sticker", title="发表情包")
        self.assertEqual([x["kind"] for x in self.log.recent()], ["image", "sticker"])

    async def test_clear_empties_buffer_and_store(self):
        await emit_trace(self.bus, trace_id="t1", session="s1", lane="think", kind="decision")
        await self.log.on_event(Event(topic=TOPIC_TRACE,
                                      payload=TraceEvent(trace_id="t2", lane="act", kind="output")))
        self.assertGreaterEqual(await self.store.clear_traces(), 1)
        self.log.clear()
        self.assertEqual(self.log.recent(), [])

    async def test_buffer_is_bounded(self):
        """环形缓冲满则丢最旧（内存不无限增长）。"""
        small = TraceLog(self.store, buffer=3)
        small.attach(EventBus())
        for i in range(5):
            await small.write(TraceEvent(trace_id=f"t{i}", lane="think", kind="decision"))
        kept = small.recent()
        self.assertEqual(len(kept), 3)
        self.assertEqual([x["seq"] for x in kept], [3, 4, 5])


# =========================================================================== 桩
class _FakeLLM:
    """可编程 LLM 替身：按脚本弹内容，供 Mind 五步驱动。"""

    def __init__(self, script: list[tuple[str, Any]]) -> None:
        self.script = script
        self.i = 0
        self.calls: list[list[Any]] = []

    async def chat(self, messages, *, tools=None, json_mode=False,
                   temperature=None, max_tokens=None, task="") -> LLMResponse:
        self.calls.append(list(messages))
        idx = min(self.i, len(self.script) - 1)
        self.i += 1
        content, tool = self.script[idx]
        return LLMResponse(content=content, tool_call=tool, model="fake")


# =========================================================================== Mind 插桩
class TestMindInstrumentation(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        # Mind 用记忆库；TraceLog 用管理库（traces 表）——两者职责不同，各就各位。
        self.mem = MemoryStore(os.path.join(self._tmp.name, "mind.db"))
        self.store = AdminStore(os.path.join(self._tmp.name, "admin.db"))
        self.bus = EventBus()
        self.log = TraceLog(self.store, buffer=50)
        self.log.attach(self.bus)
        self.builder = ContextBuilder(window_fn=lambda: 8192, ratio_fn=lambda: 0.0)

    async def asyncTearDown(self) -> None:
        await self.mem.close()
        await self.store.close()
        self._tmp.cleanup()

    def _mind(self) -> Mind:
        llm = _FakeLLM([
            ('{"action":"speak","content":"你好呀","importance":0.4,"reason":"礼貌"}', None),
        ])
        loop = AgentLoop(llm, {}, "你是助手")
        return Mind(llm, loop, store=self.mem, bus=self.bus,
                    builder=self.builder, persona_prompt="你叫小赛。")

    async def test_wake_emits_input_think_act(self):
        mind = self._mind()
        ctx = SessionContext(20)
        trigger = CoreInput(source="webchat_message", content="你好", user="owner",
                            channel="webchat", session="s1")
        result = await mind.wake(ctx, trigger)

        self.assertEqual(result.decision.action, "speak")
        items = self.log.recent()
        self.assertEqual({x["lane"] for x in items}, {"input", "think", "act"})
        self.assertEqual({x["session"] for x in items}, {"s1"})     # 全部归属本会话
        kinds = {x["kind"] for x in items}
        self.assertIn("user_msg", kinds)
        self.assertIn("decision", kinds)
        self.assertIn("output", kinds)

    async def test_sandbox_does_not_persist(self):
        """沙盒会话照常推理出输出，但不写 messages 表。"""
        mind = self._mind()
        ctx = SessionContext(20)
        trigger = CoreInput(source="webchat_message", content="你好", user="owner",
                            channel="webchat", session="s9", sandbox=True)
        result = await mind.wake(ctx, trigger)

        self.assertEqual(result.decision.action, "speak")
        self.assertEqual(await self.mem.recent_messages(10, "owner"), [])

    async def test_no_bus_still_runs(self):
        """未注入总线（阶段二老构造）也能正常跑，插桩零侵入。"""
        llm = _FakeLLM([
            ('{"action":"speak","content":"好呀","importance":0.3,"reason":"回应"}', None),
        ])
        loop = AgentLoop(llm, {}, "你是助手")
        mind = Mind(llm, loop, store=self.mem, persona_prompt="你叫小赛。")
        result = await mind.wake(SessionContext(20),
                                 CoreInput(source="cli_message", content="在吗", user="u", channel="cli"))
        self.assertEqual(result.decision.action, "speak")


if __name__ == "__main__":
    unittest.main()
