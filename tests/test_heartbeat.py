"""心跳路由测试（阶段三补丁：主动发言回到正确会话）。

背景（用户报告的缺陷）：在 webchat 里问话，第一次回复能回到 chat；
但机器人第二次「主动搭话」（心跳）时，输出被路由到了 cli。
根因：心跳触发包用「第一个渠道」（通常是 cli）当 channel，
Mind._to_output 于是继承了 cli，webchat 渠道拿不到这条主动消息。

修复：BeatLoop 记住「最近一次用户消息的来路」（_last_route），
心跳构造触发包时沿用该来路，回复便自然回到原渠道/原会话。
"""
from __future__ import annotations

import unittest
from typing import Any

from ska.heartbeat.beat import BeatLoop
from ska.kernel.bus import EventBus
from ska.kernel.context import SessionContext
from ska.kernel.event import TOPIC_INPUT, Event
from ska.kernel.packet import CoreInput, CoreOutput, Decision, MindResult
from ska.loop.sessions import SessionRegistry


# =========================================================================== 桩
class _Cfg:
    """最小 Config 替身：只提供 beat 需要的 bpm 与 throttle 小节。"""

    def __init__(self) -> None:
        self.throttle = {"active_hours": [0, 24], "proactive_max_per_min": 2,
                         "max_silent_beats": 10, "backoff_factor": 2, "send_fail_limit": 3}

    def get(self, key: str, default: Any = None) -> Any:
        return {"bpm": 6}.get(key, default)


class _Channel:
    """记录 send() 的渠道替身。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.sent: list[CoreOutput] = []

    async def start(self) -> None:
        return None

    async def send(self, out: CoreOutput) -> bool:
        self.sent.append(out)
        return True

    async def stop(self) -> None:
        return None


class _Mind:
    """把 trigger 的路由原样搬进 output（复刻 Mind._to_output 的关键语义）。"""

    def __init__(self) -> None:
        self.woke: list[CoreInput] = []

    async def wake(self, ctx: SessionContext, trigger: CoreInput) -> MindResult:
        self.woke.append(trigger)
        return MindResult(decision=Decision(action="speak"),
                          output=CoreOutput(content="嗨", channel=trigger.channel,
                                            user=trigger.user, session=trigger.session))


def _make_beat(channels: dict[str, _Channel], registry: Any = None) -> BeatLoop:
    return BeatLoop(EventBus(), _Cfg(), _Mind(), SessionContext(20), channels,
                    registry=registry if registry is not None else SessionRegistry(20))


# =========================================================================== 用例
class TestHeartbeatRouting(unittest.IsolatedAsyncioTestCase):
    def _channels(self) -> dict[str, _Channel]:
        return {"cli": _Channel("cli"), "webchat": _Channel("webchat")}

    async def test_remember_route_only_user_messages(self):
        beat = _make_beat(self._channels())
        beat._remember_route(CoreInput(source="heartbeat", content="心跳"))
        self.assertEqual(beat._last_route, {})                      # 心跳不记
        beat._remember_route(CoreInput(source="system_event", content="断线", event="x"))
        self.assertEqual(beat._last_route, {})                      # 系统事件不记
        beat._remember_route(CoreInput(source="webchat_message", content="你好",
                                       channel="webchat", user="owner", session="s7"))
        self.assertEqual(beat._last_route["channel"], "webchat")
        self.assertEqual(beat._last_route["session"], "s7")

    async def test_proactive_returns_to_webchat_not_cli(self):
        """核心回归：webchat 聊过 → 心跳主动搭话仍回 webchat，而非 cli。"""
        channels = self._channels()
        beat = _make_beat(channels)

        # 第一次：用户在 webchat 问话（经事件路径，顺带记下路由）
        msg = CoreInput(source="webchat_message", content="在吗", channel="webchat",
                        user="owner", session="s1")
        await beat._on_events([Event(topic=TOPIC_INPUT, payload=msg)])
        self.assertEqual(channels["webchat"].sent[-1].session, "s1")

        # 第二次：心跳主动搭话
        await beat._on_heartbeat()
        self.assertEqual(len(channels["webchat"].sent), 2)
        self.assertEqual(channels["cli"].sent, [])                  # 修复点：不再误路由到 cli
        self.assertEqual(channels["webchat"].sent[-1].session, "s1")

    async def test_proactive_uses_primary_without_history(self):
        """从未收过用户消息时，仍退回主渠道（向后兼容）。"""
        channels = self._channels()
        beat = _make_beat(channels)
        trigger = await beat._pick_trigger()
        self.assertEqual(trigger.channel, "cli")
        self.assertEqual(trigger.source, "heartbeat")

    async def test_last_route_carries_group_and_type(self):
        """群消息来路被完整继承（渠道/类型/群号）。"""
        channels = self._channels()
        beat = _make_beat(channels)
        msg = CoreInput(source="qq_message", content="群里说话", channel="qq",
                        user="123", message_type="group", group_id="456")
        beat._remember_route(msg)
        trigger = await beat._pick_trigger()
        self.assertEqual((trigger.channel, trigger.message_type, trigger.group_id),
                         ("qq", "group", "456"))

    async def test_whitelist_target_beats_last_route(self):
        """配置了 QQ 白名单目标时，优先走白名单（D7 行为不变）。"""
        channels = self._channels()
        channels["qq"] = _Channel("qq")
        channels["qq"].connected = True                             # type: ignore[attr-defined]

        class _Policy:
            async def targets(self, channel: str = ""):
                return [{"target_type": "private", "target_id": "888", "enabled": True,
                         "channel": "qq"}] if channel == "qq" else []

        beat = _make_beat(channels)
        beat.policy = _Policy()
        beat._remember_route(CoreInput(source="webchat_message", content="hi",
                                       channel="webchat", user="owner", session="s1"))
        trigger = await beat._pick_trigger()
        self.assertEqual(trigger.channel, "qq")                     # 白名单优先
        self.assertEqual(trigger.user, "888")

    # --------------------------------------------------------------- 自主休眠
    async def test_agent_sleep_overrides_next_timeout(self):
        """自主休眠：Agent 给出 sleep_seconds 后，下一轮等待时长被覆盖（心跳仅兜底）。"""
        beat = _make_beat(self._channels())
        beat._note_sleep(MindResult(decision=Decision(action="silence", sleep_seconds=120)))
        self.assertEqual(beat._next_sleep, 120.0)

    async def test_agent_sleep_clamped_and_zero_ignored(self):
        """sleep_seconds 上限被钳制；0/缺省不改变默认心跳。"""
        beat = _make_beat(self._channels())
        beat._note_sleep(MindResult(decision=Decision(action="silence", sleep_seconds=999999)))
        self.assertEqual(beat._next_sleep, 86400.0)      # 钳到 24 小时上限
        beat._next_sleep = 0.0
        beat._note_sleep(MindResult(decision=Decision(action="silence")))
        self.assertEqual(beat._next_sleep, 0.0)          # 缺省 → 沿用默认心跳


if __name__ == "__main__":
    unittest.main()
