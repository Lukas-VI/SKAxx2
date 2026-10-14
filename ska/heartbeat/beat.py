"""BeatLoop —— 心跳主循环。

继承老 core.py:89-124 的核心句法，并修掉三处硬伤：

    async def run(self):
        while not stop:
            fired = await bus.wait(timeout=60 / bpm)   # 原样继承「超时=心跳」的思想
            if fired:
                await self._on_events(await bus.drain())   # ← 修复：批量取空，不再丢消息
            else:
                self.beat_count += 1
                await self._on_heartbeat()                 # ← 修复：主动路径真的会说话（带三重闸）

    三重闸（只约束主动发言，用户消息永远放行）：
        ① active_hours  作息时间（跨午夜安全）
        ② 频率闸        proactive_max_per_min 滑窗
        ③ 静默退避      连续 max_silent_beats 次沉默 → 跳过若干拍（临时降频）
    另有发送熔断：连续 send_fail_limit 次发送失败则停止主动发送，等 channel_online 恢复。
"""
from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from typing import Any

from ..kernel.bus import EventBus
from ..kernel.context import SessionContext
from ..kernel.event import (TOPIC_CHANNEL_STATUS, TOPIC_OUTPUT, TOPIC_SYSTEM,
                            Event)
from ..kernel.packet import CoreInput, MindResult

logger = logging.getLogger(__name__)


class BeatLoop:
    """心跳循环：装配好之后调用 run() 进入主循环。"""

    def __init__(
        self,
        bus: EventBus,
        cfg: Any,
        mind: Any,
        ctx: SessionContext,
        channels: dict[str, Any],
        *,
        settings: Any = None,
        policy: Any = None,
    ) -> None:
        self.bus = bus
        self.cfg = cfg
        self.mind = mind
        self.ctx = ctx
        self.channels = channels
        self.settings = settings      # admin.settings.RuntimeSettings（热改参数；可为 None）
        self.policy = policy          # admin.policy.Policy（白名单校验；可为 None）

        self.beat_count = 0
        self._stop = False

        # 主动发言默认走第一个渠道
        self._primary = next(iter(channels), "cli")

        # 三重闸参数（先用 config.yaml 的初值；启用后台后由 _refresh 热读覆盖层）
        self._load_from(cfg)
        self._refresh()

        # 运行时状态
        self._proactive_ts: deque[float] = deque()   # 主动发言时间戳（滑窗）
        self._silent_beats = 0                       # 连续沉默拍数
        self._skip_beats = 0                         # 退避期间还要跳过的拍数
        self._send_fails = 0                         # 连续发送失败次数
        self._circuit_open = False                   # 熔断开关

    # ------------------------------------------------------------------ 配置热读
    def _load_from(self, cfg: Any) -> None:
        """从 config.yaml（Config）载入 bpm 与三重闸参数。"""
        self.bpm = max(int(cfg.get("bpm", 6)), 1)
        th = cfg.throttle
        self.active_hours = th.get("active_hours", [0, 24])
        self.max_per_min = int(th.get("proactive_max_per_min", 2))
        self.max_silent = int(th.get("max_silent_beats", 10))
        self.backoff_factor = max(int(th.get("backoff_factor", 2)), 1)
        self.send_fail_limit = int(th.get("send_fail_limit", 3))

    def _refresh(self) -> None:
        """每拍重读「后台覆盖层」，让改 bpm/三重闸无需重启进程（settings 为 None 时无操作）。"""
        if self.settings is None:
            return
        try:
            self.bpm = max(int(self.settings.get("bpm", self.bpm) or self.bpm), 1)
            th = self.settings.section("throttle")
            self.active_hours = th.get("active_hours", self.active_hours)
            self.max_per_min = int(th.get("proactive_max_per_min", self.max_per_min))
            self.max_silent = int(th.get("max_silent_beats", self.max_silent))
            self.backoff_factor = max(int(th.get("backoff_factor", self.backoff_factor)), 1)
            self.send_fail_limit = int(th.get("send_fail_limit", self.send_fail_limit))
        except (TypeError, ValueError):
            logger.warning("心跳参数热读失败，沿用旧值", exc_info=True)

    def current_bpm(self) -> int:
        """当前生效的 bpm（供后台「运行状态」页展示；热更新后即为新值）。"""
        return self.bpm

    # ------------------------------------------------------------------ 主循环
    async def run(self) -> None:
        """进入主循环，直到 request_stop() 被调用。"""
        logger.info("核心启动：bpm=%d（每拍 %.1fs）", self.bpm, 60 / self.bpm)
        while not self._stop:
            self._refresh()                              # 每拍热读 bpm/闸门参数
            fired = await self.bus.wait(timeout=60 / self.bpm)
            if self._stop:
                break
            if fired:
                await self._on_events(await self.bus.drain())
            else:
                self.beat_count += 1
                logger.debug("[%d] 心跳", self.beat_count)
                await self._on_heartbeat()
        logger.info("核心已停止（累计心跳 %d 次）", self.beat_count)

    def request_stop(self) -> None:
        """请求停止（信号处理器在事件循环线程里调用）。"""
        self._stop = True
        self.bus.wakeup()   # 让正在等待的 wait() 立刻返回

    # ------------------------------------------------------------------ 事件路径
    async def _on_events(self, events: list[Event]) -> None:
        """处理一批输入事件（drain 出来的）。"""
        for ev in events:
            if ev.topic == TOPIC_SYSTEM:
                payload = ev.payload
                if getattr(payload, "event", "") == "shutdown":
                    logger.info("收到 shutdown 事件，准备退出")
                    self.request_stop()
                    return
            if ev.topic == TOPIC_CHANNEL_STATUS:
                await self._on_channel_status(ev)
                continue
            if not ev.is_input():
                continue
            # 单个事件处理失败不能拖垮整个心跳循环（日志留痕后继续）
            try:
                result = await self.mind.wake(self.ctx, ev.payload)
            except Exception:  # noqa: BLE001
                logger.exception("处理事件失败，已跳过：topic=%s", ev.topic)
                continue
            await self._deliver(result, proactive=False)

    async def _on_channel_status(self, ev: Event) -> None:
        """渠道断线/重连。

        重连（channel_online）会喂给 Mind，让它有机会说一句「我回来了」，
        同时清除发送熔断；断线只记日志并打开熔断（避免断线期间空耗 LLM）。
        """
        payload = ev.payload
        topic_event = getattr(payload, "event", "")
        if topic_event == "channel_online":
            self._circuit_open = False
            self._send_fails = 0
            logger.info("渠道恢复在线：%s", getattr(payload, "content", ""))
            result = await self.mind.wake(self.ctx, payload)
            await self._deliver(result, proactive=False)
        else:
            self._circuit_open = True
            logger.warning("渠道断开：%s", getattr(payload, "content", ""))

    # ------------------------------------------------------------------ 主动路径
    async def _on_heartbeat(self) -> None:
        """一次心跳超时：三重闸通过后，问 Mind 要不要主动说话。"""
        if self._skip_beats > 0:            # 退避中：跳过若干拍
            self._skip_beats -= 1
            return
        if not self._gate_open():
            return

        trigger = await self._pick_trigger()
        result = await self.mind.wake(self.ctx, trigger)

        if result.output is not None:
            await self._deliver(result, proactive=True)
        else:
            # 沉默：累计到阈值就临时降频
            self._silent_beats += 1
            if self._silent_beats >= self.max_silent:
                self._silent_beats = 0
                self._skip_beats = self.backoff_factor
                logger.info("连续沉默 %d 拍，接下来跳过 %d 拍以降低频率",
                            self.max_silent, self.backoff_factor)

    async def _pick_trigger(self) -> CoreInput:
        """构造本次心跳的触发输入：优先从「可发布白名单」选一个真实目标。

        D7：白名单只在配置了目标时生效。未配置（或 QQ 未在线）时，
        退回阶段一行为——用主渠道发一条私聊式主动消息。
        """
        target = await self._pick_target()
        if target is not None:
            return target
        return CoreInput(source="heartbeat", content="心跳唤醒", channel=self._primary)

    async def _pick_target(self) -> CoreInput | None:
        """从白名单挑选一个可主动发言的目标（群或好友）；无可选项时返回 None。"""
        if self.policy is None:
            return None
        qq = self.channels.get("qq")
        if qq is None or not getattr(qq, "connected", False):
            return None                       # QQ 不在线时不走 QQ 主动路径
        targets = await self.policy.targets("qq")
        if not targets:
            return None                       # 未配置白名单 → 退回默认
        t = targets[0]
        if t.get("target_type") == "group":
            return CoreInput(source="heartbeat", content="心跳唤醒", channel="qq",
                             message_type="group", group_id=str(t.get("target_id", "")))
        return CoreInput(source="heartbeat", content="心跳唤醒", channel="qq",
                         message_type="private", user=str(t.get("target_id", "")))

    def _gate_open(self) -> bool:
        """三重闸：作息 → 熔断 → 频率。全过才允许主动发言。"""
        if not self._in_active_hours():
            return False
        if self._circuit_open:
            return False
        return self._under_rate_limit()

    def _in_active_hours(self) -> bool:
        """作息判定，跨午夜安全：[8, 23) 表示 8 点到 23 点。"""
        try:
            start, end = int(self.active_hours[0]), int(self.active_hours[1])
        except (IndexError, TypeError, ValueError):
            return True
        hour = datetime.now().hour
        if start <= end:
            return start <= hour < end
        # 跨午夜，如 [23, 8)
        return hour >= start or hour < end

    def _under_rate_limit(self) -> bool:
        """频率闸：最近 60 秒内主动发言次数 < 上限。"""
        now = time.time()
        while self._proactive_ts and now - self._proactive_ts[0] > 60:
            self._proactive_ts.popleft()
        return len(self._proactive_ts) < self.max_per_min

    # ------------------------------------------------------------------ 交付
    async def _deliver(self, result: MindResult, *, proactive: bool) -> None:
        """把 Mind 的输出交给渠道发送（先过 output 事件，让 filter 等插件净化）。"""
        out = result.output
        if out is None or not out.is_visible():
            return

        # 主动发言二次校验白名单（D7）：即便 Mind 产出了输出，也不越白名单乱发
        if proactive and self.policy is not None:
            target_type = "group" if out.target == "group_msg" else "user"
            target_id = out.group_id if out.target == "group_msg" else out.user
            if not await self.policy.can_publish(out.channel, target_type, target_id):
                logger.info("目标 %s/%s 不在可发布白名单内，跳过本次主动发送",
                            target_type, target_id)
                return

        # 信号类事件：只通知订阅者（filter 会在此原地清洗 content）
        await self.bus.emit(TOPIC_OUTPUT, out)

        channel = self.channels.get(out.channel) or self.channels.get(self._primary)
        if channel is None:
            logger.warning("没有可用渠道，丢弃输出")
            return
        ok = await channel.send(out)
        if ok:
            self._send_fails = 0
            if proactive:
                self._proactive_ts.append(time.time())
        else:
            self._send_fails += 1
            if self._send_fails >= self.send_fail_limit:
                self._circuit_open = True
                logger.warning("发送连续失败 %d 次，熔断主动发送", self._send_fails)