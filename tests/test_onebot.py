"""OneBot v11 客户端与 QQ 渠道解析测试（阶段二新增）。

覆盖：
    * OneBotClient：envelope 解包、retcode 非 0 报错、网络错误转 OneBotError、
      send_group_msg 的 URL/参数（用 httpx.MockTransport 拦截，不发真实请求）；
    * QQChannel：OneBot 事件 → CoreInput 的字段映射与文本抽取（array / string）。
"""
from __future__ import annotations

import json
import unittest

import httpx

from ska.io.base import ChannelDeps
from ska.io.onebot import OneBotClient, OneBotError
from ska.io.qq import QQChannel
from ska.kernel.bus import EventBus


class _StubCfg:
    """QQChannel 只用 deps.cfg.section("qq")，给个最小替身即可。"""

    def section(self, name):
        return {}


def _client(handler) -> OneBotClient:
    return OneBotClient("http://127.0.0.1:3000", access_token="tok",
                        transport=httpx.MockTransport(handler))


class TestOneBotClient(unittest.IsolatedAsyncioTestCase):
    async def test_send_group_msg_ok(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("authorization")
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={"status": "ok", "retcode": 0,
                                             "data": {"message_id": 42}})

        client = _client(handler)
        data = await client.send_group_msg(999, "晚上好")
        await client.aclose()

        self.assertTrue(captured["url"].endswith("/send_group_msg"))
        self.assertEqual(captured["auth"], "Bearer tok")
        self.assertEqual(captured["body"]["group_id"], 999)
        self.assertEqual(captured["body"]["message"], [{"type": "text", "data": {"text": "晚上好"}}])
        self.assertEqual(data["message_id"], 42)

    async def test_retcode_nonzero_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "failed", "retcode": 100,
                                             "message": "无权限"})

        client = _client(handler)
        with self.assertRaises(OneBotError):
            await client.get_login_info()
        await client.aclose()

    async def test_network_error_raises_onebot_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("连不上")

        client = _client(handler)
        with self.assertRaises(OneBotError):
            await client.get_status()
        await client.aclose()

    async def test_missing_api_base(self):
        client = OneBotClient("")
        with self.assertRaises(OneBotError):
            await client.get_login_info()

    async def test_list_result_unwrapped(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "ok", "retcode": 0,
                                             "data": [{"group_id": 1, "group_name": "群1"}]})

        client = _client(handler)
        groups = await client.get_group_list()
        await client.aclose()
        self.assertEqual(groups[0]["group_name"], "群1")


class TestQQChannelParsing(unittest.IsolatedAsyncioTestCase):
    def _channel(self) -> QQChannel:
        return QQChannel(ChannelDeps(bus=EventBus(), cfg=_StubCfg()))

    def test_group_array_message(self):
        """array 格式群消息：抽文本、@某人、[图片]，并映射群/用户字段。"""
        ch = self._channel()
        packet = ch._to_input({
            "post_type": "message", "message_type": "group", "group_id": 999,
            "user_id": 10001, "time": 1700000000,
            "sender": {"card": "小明", "nickname": "小明昵称"},
            "message": [
                {"type": "text", "data": {"text": "你好 "}},
                {"type": "at", "data": {"qq": "12345"}},
                {"type": "image", "data": {"file": "x.jpg"}},
            ],
        })
        self.assertEqual(packet.source, "qq_message")
        self.assertEqual(packet.channel, "qq")
        self.assertEqual(packet.message_type, "group")
        self.assertEqual(packet.group_id, "999")
        self.assertEqual(packet.user, "10001")
        self.assertEqual(packet.user_name, "小明")          # card 优先
        self.assertEqual(packet.content, "你好 @12345[图片]")

    def test_private_string_message(self):
        """string 格式私聊消息：直接采用 message 文本，group_id 为空。"""
        ch = self._channel()
        packet = ch._to_input({
            "post_type": "message", "message_type": "private",
            "user_id": 10002, "sender": {"nickname": "小红"},
            "message": "在吗",
        })
        self.assertEqual(packet.message_type, "private")
        self.assertEqual(packet.group_id, "")
        self.assertEqual(packet.user_name, "小红")
        self.assertEqual(packet.content, "在吗")

    def test_raw_message_fallback(self):
        """message 缺失时退回 raw_message（含 CQ 码）。"""
        ch = self._channel()
        packet = ch._to_input({"post_type": "message", "message_type": "private",
                               "user_id": 1, "raw_message": "[CQ:face,id=1]"})
        self.assertEqual(packet.content, "[CQ:face,id=1]")

    async def test_on_event_rejects_non_message(self):
        ch = self._channel()
        self.assertFalse(await ch.on_event({"post_type": "meta_event"}))
        self.assertFalse(await ch.on_event({"post_type": "notice"}))
        self.assertFalse(await ch.on_event("不是字典"))            # 类型不对

    async def test_on_event_ignores_self(self):
        """机器人自己发的消息会被忽略（防自循环）。"""
        ch = self._channel()
        ch.self_id = "10001"
        self.assertFalse(await ch.on_event({"post_type": "message", "message_type": "group",
                                            "user_id": 10001, "message": "我自己说的"}))


if __name__ == "__main__":
    unittest.main()