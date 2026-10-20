"""在线聊天测试（阶段三 · 方案 §2）。

覆盖三块：
    * SessionRegistry —— 多会话窗口隔离 + 旧渠道默认会话向后兼容；
    * ChatHub —— 会话 CRUD、投递 CoreInput（含沙盒标记）、出站增量 poll；
    * WebChatChannel —— send() 把回复分派回对应会话队列。

用真实 EventBus（无外部依赖），数据库一律建在临时目录。
"""
from __future__ import annotations

import os
import tempfile
import unittest

from ska.io.base import ChannelDeps
from ska.io.webchat import WebChatChannel
from ska.kernel.bus import EventBus
from ska.kernel.packet import CoreInput, CoreOutput, TraceEvent
from ska.loop.sessions import SessionRegistry
from ska.web.chat import ChatHub


class _StubCfg:
    """Config 最小替身：只提供 section()（渠道读 webchat 小节用）。"""

    def __init__(self, data: dict | None = None) -> None:
        self._d = data or {}

    def section(self, name: str) -> dict:
        value = self._d.get(name, {})
        return value if isinstance(value, dict) else {}

    def get(self, key, default=None):
        return self._d.get(key, default)


# =========================================================================== 会话登记表
class TestSessionRegistry(unittest.TestCase):
    def test_old_channels_share_default_session(self):
        """旧渠道不传 session → key 恒定，行为与阶段二一致（向后兼容）。"""
        reg = SessionRegistry(window=10)
        a = CoreInput(source="cli_message", content="hi", user="u", channel="cli")
        self.assertEqual(SessionRegistry.key_for(a), "cli:u:default")
        self.assertIs(reg.context_for(a), reg.context_for(a))       # 同键复用同一窗口

    def test_multi_session_isolated_windows(self):
        """同一身份开两会话：窗口隔离，长期记忆/权限由 user 维度共享（此处只验窗口）。"""
        reg = SessionRegistry(window=10)
        s1 = CoreInput(source="webchat_message", content="a", user="owner",
                       channel="webchat", session="s1")
        s2 = CoreInput(source="webchat_message", content="b", user="owner",
                       channel="webchat", session="s2")
        c1, c2 = reg.context_for(s1), reg.context_for(s2)
        self.assertIsNot(c1, c2)
        c1.add("user", "只进 s1")
        self.assertEqual(len(c2), 0)                               # s2 不受影响
        self.assertEqual(reg.stats()["count"], 2)
        self.assertTrue(reg.drop(SessionRegistry.key_for(s1)))
        self.assertIsNone(reg.get(SessionRegistry.key_for(s1)))


# =========================================================================== ChatHub
class TestChatHub(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.bus = EventBus()
        self.hub = ChatHub(self.bus, default_identity="owner", default_persist=True)

    async def test_create_defaults(self):
        s = self.hub.create()
        self.assertEqual(s.identity, "owner")
        self.assertTrue(s.persist)
        self.assertFalse(s.info()["sandbox"])
        # 指定 id 与标题
        s2 = self.hub.create(session_id="abc", title="标题", identity="guest", persist=False)
        self.assertEqual(s2.id, "abc")
        self.assertEqual(s2.title, "标题")
        self.assertEqual(s2.identity, "guest")
        self.assertTrue(s2.info()["sandbox"])
        with self.assertRaises(ValueError):
            self.hub.create(session_id="abc")                      # 重复 id 拒绝

    async def test_submit_publishes_core_input(self):
        s = self.hub.create()
        r = await self.hub.submit(s.id, "你好")
        self.assertEqual(r["message"]["role"], "user")
        self.assertEqual(r["session"]["pending"], True)            # 等回复
        self.assertEqual(s.title, "你好")                          # 首条消息当标题

        events = await self.bus.drain()
        self.assertEqual(len(events), 1)
        pkt = events[0].payload
        self.assertIsInstance(pkt, CoreInput)
        self.assertEqual(pkt.source, "webchat_message")
        self.assertEqual(pkt.channel, "webchat")
        self.assertEqual(pkt.session, s.id)
        self.assertEqual(pkt.user, "owner")
        self.assertFalse(pkt.sandbox)

    async def test_sandbox_session_marks_packet(self):
        s = self.hub.create(persist=False)
        await self.hub.submit(s.id, "沙盒里说的话")
        pkt = (await self.bus.drain())[-1].payload
        self.assertTrue(pkt.sandbox)                               # 不落库，但照常推理

    async def test_submit_empty_rejected(self):
        s = self.hub.create()
        with self.assertRaises(ValueError):
            await self.hub.submit(s.id, "   ")

    async def test_push_then_poll_incremental(self):
        s = self.hub.create()
        await self.hub.submit(s.id, "第一句")
        ok = self.hub.push(s.id, CoreOutput(content="回复", channel="webchat", session=s.id))
        self.assertTrue(ok)

        full = self.hub.poll(s.id, after=0)
        self.assertEqual([m["role"] for m in full["messages"]], ["user", "assistant"])
        self.assertFalse(full["session"]["pending"])               # push 复位 pending

        inc = self.hub.poll(s.id, after=1)                          # 只取 seq>1
        self.assertEqual(len(inc["messages"]), 1)
        self.assertEqual(inc["messages"][0]["content"], "回复")
        self.assertEqual(inc["last_seq"], 2)

    async def test_push_without_session_falls_back_to_latest(self):
        a = self.hub.create()
        b = self.hub.create()
        _ = a
        self.hub.push("", CoreOutput(content="没有会话的输出"))       # 落到最近活跃会话
        self.assertEqual(self.hub.poll(b.id)["messages"][-1]["content"], "没有会话的输出")

    async def test_update_and_delete(self):
        s = self.hub.create()
        info = self.hub.update(s.id, title="改名", persist=False, identity="guest")
        self.assertEqual(info["title"], "改名")
        self.assertFalse(info["persist"])
        self.assertEqual(info["identity"], "guest")
        self.assertTrue(self.hub.delete(s.id))
        self.assertFalse(self.hub.delete(s.id))                     # 幂等
        with self.assertRaises(ValueError):
            self.hub.poll(s.id)

    async def test_on_trace_updates_tokens(self):
        s = self.hub.create()
        self.hub.on_trace(TraceEvent(session=s.id, lane="think", kind="decision", tokens=123))
        self.assertEqual(self.hub.get(s.id).tokens, 123)
        # 未知会话的事件被忽略（不应报错）
        self.hub.on_trace(TraceEvent(session="不存在", tokens=999))

    async def test_list_orders_by_recent(self):
        a = self.hub.create(session_id="one")
        b = self.hub.create(session_id="two")
        self.hub.update(b.id, title="最新的")
        ids = [x["id"] for x in self.hub.list()]
        self.assertEqual(ids[0], b.id)
        self.assertIn(a.id, ids)


# =========================================================================== 渠道
class TestWebChatChannel(unittest.IsolatedAsyncioTestCase):
    async def test_send_dispatches_to_hub(self):
        bus = EventBus()
        hub = ChatHub(bus)
        deps = ChannelDeps(bus=bus, cfg=_StubCfg({"webchat": {"enabled": True, "identity": "owner"}}),
                           extras={"chat_hub": hub})
        channel = WebChatChannel(deps)
        await channel.start()
        self.assertTrue(channel.connected)

        s = hub.create()
        ok = await channel.send(CoreOutput(content="嗨", channel="webchat", session=s.id))
        self.assertTrue(ok)
        self.assertEqual(hub.poll(s.id)["messages"][-1]["content"], "嗨")

    async def test_send_without_hub_returns_false(self):
        """未注入 ChatHub → 退化为丢弃（核心仍可启动），不抛异常。"""
        deps = ChannelDeps(bus=EventBus(), cfg=_StubCfg({}), extras={})
        channel = WebChatChannel(deps)
        self.assertFalse(await channel.send(CoreOutput(content="x")))
        self.assertFalse(channel.connected)


if __name__ == "__main__":
    unittest.main()
