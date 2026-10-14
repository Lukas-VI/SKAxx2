"""内核数据包测试：来源显式声明、路由目标、决策结构校验。"""
from __future__ import annotations

import unittest

from pydantic import ValidationError

from ska.kernel.packet import (AgentResult, CoreInput, CoreOutput, Decision,
                               MindResult, ToolCall)


class TestCoreInput(unittest.TestCase):
    def test_source_is_explicit(self):
        """来源由构造方显式声明（老代码是靠字段猜来源）。"""
        pkt = CoreInput(source="qq_message", content="在吗", user="123",
                        user_name="小明", message_type="group", group_id="456")
        self.assertTrue(pkt.is_user_message())
        self.assertEqual(pkt.group_id, "456")

    def test_defaults(self):
        pkt = CoreInput(source="cli_message", content="hi")
        self.assertEqual(pkt.channel, "cli")
        self.assertEqual(pkt.message_type, "private")

    def test_system_event_is_not_user_message(self):
        pkt = CoreInput(source="system_event", event="shutdown")
        self.assertFalse(pkt.is_user_message())

    def test_heartbeat_is_not_user_message(self):
        pkt = CoreInput(source="heartbeat", task="morning_greeting")
        self.assertFalse(pkt.is_user_message())

    def test_invalid_source_rejected(self):
        """非法来源直接报错，不再静默降级为 unknown。"""
        with self.assertRaises(ValidationError):
            CoreInput(source="guess_me")


class TestCoreOutput(unittest.TestCase):
    def test_internal_invisible(self):
        self.assertFalse(CoreOutput(content="心里话", target="internal").is_visible())

    def test_visible_targets(self):
        self.assertTrue(CoreOutput(content="嗨", target="private_msg").is_visible())
        self.assertTrue(CoreOutput(content="嗨", target="group_msg").is_visible())

    def test_invalid_target_rejected(self):
        with self.assertRaises(ValidationError):
            CoreOutput(content="x", target="broadcast")


class TestDecision(unittest.TestCase):
    def test_tool_call_shape(self):
        d = Decision(action="tool", tool=ToolCall(name="time", args={"fmt": "%H"}))
        self.assertIsNotNone(d.tool)
        self.assertEqual(d.tool.name, "time")

    def test_invalid_action_rejected(self):
        with self.assertRaises(ValidationError):
            Decision(action="dance")

    def test_defaults(self):
        d = Decision(action="silence")
        self.assertEqual(d.importance, 0.5)
        self.assertIsNone(d.tool)


class TestMindResult(unittest.TestCase):
    def test_assembles(self):
        mr = MindResult(decision=Decision(action="silence"))
        self.assertIsNone(mr.output)
        self.assertIsNone(mr.agent)

    def test_with_agent_result(self):
        mr = MindResult(
            decision=Decision(action="tool"),
            agent=AgentResult(final="现在是三点"),
            monologue="他心里想……",
        )
        self.assertEqual(mr.agent.final, "现在是三点")
        self.assertFalse(mr.agent.stopped)


if __name__ == "__main__":
    unittest.main()