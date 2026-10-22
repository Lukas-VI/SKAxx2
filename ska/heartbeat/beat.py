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

    自主休眠（阶段三增强）：Mind 决策可携带 sleep_seconds，用来覆盖默认心跳间隔——
        Agent 能自己决定「接下来睡多久」，但任何输入 / 插件事件都会立即把它唤醒，
        于是心跳从「唯一的节拍源」降级为「兜底节拍源」（见 _note_sleep / run）。
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

# Agent 自主休眠的上限（秒）：与 mind.MAX_SLEEP_SECONDS 同量级（此处独立定义，
# 让 heartbeat 层不依赖 loop 层；24 小时作为兜底，防止把心跳彻底停死）。
MAX_SLEEP_SECONDS = 86400.0

# 会被「对话窗口」合并的消息来源（用户真人消息；心跳/系统事件不合并）
USER_SOURCES = frozenset({"cli_message", "qq_message", "webchat_message"})


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
        registry: Any = None,
    ) -> None:
        self.bus = bus
        self.cfg = cfg
        self.mind = mind
        self.ctx = ctx
        self.channels = channels
        self.settings = settings      # admin.settings.RuntimeSettings（热改参数；可为 None）
        self.policy = policy          # admin.policy.Policy（白名单校验；可为 None）
        self.registry = registry      # loop.sessions.SessionRegistry（阶段三多会话；可为 None）

        self.beat_count = 0
        self._stop = False
        # Agent 自主休眠（阶段三增强）：若非 0，则用它替代默认心跳间隔作为下一次 wait 的超时。
        # 只在「处理完一轮之后」生效——期间任何输入/插件事件都会让 bus.wait 立即返回，
        # 于是达到了「Agent 自行决定睡多久，但仍可被事件随时唤醒」的效果（心跳仅作兜底）。
        self._next_sleep = 0.0

        # 主动发言默认走第一个渠道（无历史来路时的兜底）
        self._primary = next(iter(channels), "cli")
        # 最近一次「用户消息」的来路（渠道/用户/会话/群）。主动发言本身没有来路，
        # 记住它，才能让机器人主动搭话回到「刚聊过的那个会话」，而不是第一个渠道。
        self._last_route: dict[str, Any] = {}

        # 三重闸参数（先用 config.yaml 的初值；启用后台后由 _refresh 热读覆盖层）
        self._load_from(cfg)
        # 对话窗口（消息批处理）参数：默认开启，一批最多 12 条
        self.batch_enabled = True
        self.batch_max = 12
        self._load_dialog(cfg)
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

    def _load_dialog(self, cfg: Any) -> None:
        """从 config.yaml 的 dialog 小节载入对话窗口参数（缺 section 的替身也不报错）。"""
        try:
            sec = cfg.section("dialog")
        except AttributeError:                 # 测试替身只有 get/throttle
            sec = {}
        self.batch_enabled = bool(sec.get("batch_enabled", self.batch_enabled))
        try:
            self.batch_max = max(int(sec.get("batch_max_messages", self.batch_max) or 0), 1)
        except (TypeError, ValueError):
            pass

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
            # 对话窗口同样支持后台热改（改完立即按新上限合批）
            self._load_dialog(self.settings)
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
            # 下一次等待时长：Agent 指定了自主休眠就用它，否则用默认心跳间隔。
            # 读完后立刻清零——避免这个值被连续几轮沿用（它只对「紧接着的下一轮」有效）。
            timeout = self._next_sleep if self._next_sleep > 0 else 60 / self.bpm
            self._next_sleep = 0.0
            fired = await self.bus.wait(timeout=timeout)
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
    def _ctx_for(self, payload: Any) -> SessionContext:
        """按输入包选出对应会话的工作记忆窗口（阶段三多会话）。

        装配了 SessionRegistry 时按 (channel:user:session) 路由 —— 于是「怎么通用
        老渠道不传 session」自动落默认会话，行为与阶段二完全一致；
        未装配时退回构造时传入的单例 ctx（保持向后兼容）。
        """
        if self.registry is not None:
            try:
                return self.registry.context_for(payload)
            except Exception:  # noqa: BLE001 —— 路由失败不能丢消息
                logger.exception("会话路由失败，回退默认 ctx")
        return self.ctx

    def _remember_route(self, payload: Any) -> None:
        """记住最近一次「用户消息」的来路（渠道 / 用户 / 会话 / 群）。

        主动发言（心跳）本身没有来路；若不记住，_pick_trigger 就会退回「第一个渠道」
        （通常是 cli），于是「在 webchat 聊完，机器人主动搭话却发到了 cli」。
        记下来路后，_pick_trigger 便可构造一个「同渠道、同会话」的唤醒，
        回复经由该渠道自然回到原会话（webchat 渠道按 output.session 投递）。
        """
        if str(getattr(payload, "source", "")) not in (
                "cli_message", "qq_message", "webchat_message"):
            return
        self._last_route = {
            "channel": str(getattr(payload, "channel", "") or ""),
            "user": str(getattr(payload, "user", "") or ""),
            "user_name": str(getattr(payload, "user_name", "") or ""),
            "session": str(getattr(payload, "session", "") or ""),
            "message_type": str(getattr(payload, "message_type", "private") or "private"),
            "group_id": str(getattr(payload, "group_id", "") or ""),
        }

    def _note_sleep(self, result: MindResult) -> None:
        """读取 Agent 在本轮决策里给出的自主休眠时长，记为「下一轮等待时长」。

        语义：sleep_seconds>0 → 覆盖默认心跳节奏，接下来这么久内不主动发声；
        期间只要有新消息 / 插件事件，bus.wait 会立即返回并把它唤醒（事件优先级最高）。
        0（或缺失）→ 什么都不做，沿用默认心跳。
        """
        try:
            seconds = float(getattr(result.decision, "sleep_seconds", 0) or 0)
        except (TypeError, ValueError):
            seconds = 0.0
        if seconds > 0:
            self._next_sleep = min(seconds, MAX_SLEEP_SECONDS)
            logger.info("Agent 自主休眠 %.0fs（覆盖心跳；期间事件可立即唤醒）", self._next_sleep)

    async def _on_events(self, events: list[Event]) -> None:
        """处理一批输入事件（drain 出来的）。

        对话窗口机制（本阶段新增）：
            模型推理一条消息往往要好几秒，这期间同一个聊天里可能又来了好几条。
            旧实现逐条 wake → 逐条回复，于是在高频群里永远在「补一句旧话」。
            现在改为：先把这批事件按「聊天」分组，同一聊天的多条消息（含图片/表情）
            合并成一个 CoreInput，只推理、只回复一次；多出来的文字按行拼接，
            图片则并进 images 列表一起交给视觉分批解析（Mind 内部有数量上限）。
        """
        inputs: list[Any] = []
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
            inputs.append(ev.payload)

        for trigger in self._batch_inputs(inputs):
            # 单个事件处理失败不能拖垮整个心跳循环（日志留痕后继续）
            try:
                self._remember_route(trigger)        # 先记下来路，供主动发言回到正确会话
                result = await self.mind.wake(self._ctx_for(trigger), trigger)
            except Exception:  # noqa: BLE001
                logger.exception("处理事件失败，已跳过：source=%s",
                                 getattr(trigger, "source", ""))
                continue
            self._note_sleep(result)                  # Agent 可能要求自主休眠
            # scheduler 等插件可在 raw["_targets"] 里附带投递目标（可多个）
            await self._deliver(result, proactive=False,
                                targets=self._event_targets(trigger))

    # ------------------------------------------------------------------ 对话窗口（批处理）
    def _batch_inputs(self, payloads: list[Any]) -> list[Any]:
        """把「同一聊天窗口里堆积的多条消息」合并成一批，其余原样通过。

        分组键：
            * 群聊 → group:<渠道>:<群号>（同一群不同人说话也属于同一窗口）；
            * 私聊 / webchat → 会话键（渠道+用户+会话），与 SessionRegistry 一致。
        非用户消息（心跳、系统事件）不参与合并，各自独立成一次唤醒。
        """
        if not self.batch_enabled or len(payloads) <= 1:
            return payloads

        groups: dict[str, list[Any]] = {}
        order: list[str] = []
        for payload in payloads:
            key = self._chat_key(payload)
            if not key:                              # 非用户消息：独立处理
                groups[f"\0{id(payload)}"] = [payload]
                order.append(f"\0{id(payload)}")
                continue
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(payload)
        return [self._merge_batch(groups[k]) for k in order]

    def _chat_key(self, payload: Any) -> str:
        """算出一条用户消息所属的「聊天窗口」；非用户消息返回空串。"""
        source = str(getattr(payload, "source", "") or "")
        if source not in USER_SOURCES:
            return ""
        channel = str(getattr(payload, "channel", "") or "")
        if str(getattr(payload, "message_type", "") or "") == "group":
            return f"group:{channel}:{getattr(payload, 'group_id', '')}"
        if self.registry is not None:
            return self.registry.key_for(payload)
        return (f"{channel}:{getattr(payload, 'user', '')}:"
                f"{getattr(payload, 'session', '')}")

    def _merge_batch(self, payloads: list[Any]) -> Any:
        """把同一窗口的多条消息压成一条 CoreInput（最后一条的来路作为回复目标）。"""
        if len(payloads) == 1:
            return payloads[0]
        last = payloads[-1]
        # 高频刷屏时永远追不上：只保留最近 N 条，更早的压成一句提示（而不是无限补课）
        kept = payloads[-self.batch_max:]
        dropped = len(payloads) - len(kept)
        lines: list[str] = []
        images: list[dict[str, Any]] = []
        for payload in kept:
            name = str(getattr(payload, "user_name", "") or getattr(payload, "user", "") or "")
            text = str(getattr(payload, "content", "") or "").strip()
            lines.append(f"{name}：{text}" if name else text)   # 标明是谁说的，群聊不乱
            images.extend(list(getattr(payload, "images", None) or []))
        content = "\n".join(lines)
        if dropped:
            content = f"（更早还有 {dropped} 条消息未展开）\n{content}"
        logger.info("对话窗口合并：%d 条消息 → 1 批（渠道=%s，用户=%s）",
                    len(payloads), getattr(last, "channel", ""),
                    getattr(last, "user", ""))
        # 用最后一条的来路（渠道/会话/群）作为回复目标，其余字段保留触发源语义；
        # raw 里记一个 _batch 标记，供 Mind 生成提示词时知道这是「一次性多句」。
        raw = dict(getattr(last, "raw", None) or {})
        raw["_batch"] = len(payloads)
        try:
            return last.model_copy(update={"content": content, "images": images, "raw": raw})
        except AttributeError:                       # 测试替身没有 model_copy
            payload = dict(last) if isinstance(last, dict) else {}
            payload.update({"content": content, "images": images, "raw": raw})
            return payload

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
            result = await self.mind.wake(self._ctx_for(payload), payload)
            self._note_sleep(result)
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
        result = await self.mind.wake(self._ctx_for(trigger), trigger)
        self._note_sleep(result)

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
        """构造本次心跳的触发输入。

        优先级（自上而下）：
            ① 白名单目标（D7：仅 QQ，配置了才走）；
            ② 最近一次用户消息的来路 —— 让主动发言回到「刚聊过的那个会话」
               （webchat 多开时尤其重要：不经此步会退回主渠道 cli）；
            ③ 兜底：主渠道（第一个渠道，通常 cli）私聊式主动消息。
        """
        target = await self._pick_target()
        if target is not None:
            return target
        route = self._last_route
        if route.get("channel"):
            return CoreInput(source="heartbeat", content="心跳唤醒",
                             channel=route["channel"],
                             user=route.get("user", ""),
                             user_name=route.get("user_name", ""),
                             session=route.get("session", ""),
                             message_type=route.get("message_type", "private"),
                             group_id=route.get("group_id", ""))
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
    @staticmethod
    def _event_targets(payload: Any) -> list[dict[str, Any]]:
        """读取插件在输入包 raw["_targets"] 里声明的投递目标（没有则为空）。

        约定：目标项形如 {"channel": "qq", "target_type": "group", "target_id": "123"}；
        scheduler 用于「日程触发后发到多个通道」，其余来源不会设置它。
        """
        raw = getattr(payload, "raw", None) or {}
        if not isinstance(raw, dict):
            return []
        targets = raw.get("_targets")
        if not isinstance(targets, list):
            return []
        return [t for t in targets if isinstance(t, dict)]

    @staticmethod
    def _fanout(out: Any, targets: list[dict[str, Any]] | None) -> list[Any]:
        """把一份输出复制成「每个目标一份」；没有目标就原样返回。"""
        items: list[Any] = []
        seen: set[tuple[str, str, str]] = set()
        for target in targets or []:
            channel = str(target.get("channel") or "").strip()
            target_type = str(target.get("target_type") or "group").strip()
            target_id = str(target.get("target_id") or "").strip()
            if not channel or not target_id:
                continue
            key = (channel, target_type, target_id)
            if key in seen:                   # 重复目标只发一次
                continue
            seen.add(key)
            is_group = target_type == "group"
            items.append(out.model_copy(update={
                "channel": channel,
                # group_msg 只有 QQ 渠道支持；其它渠道按私聊投递
                "target": "group_msg" if (is_group and channel == "qq") else "private_msg",
                "group_id": target_id if is_group else "",
                "user": "" if is_group else target_id,
            }))
        return items or [out]

    async def _deliver(self, result: MindResult, *, proactive: bool,
                       targets: list[dict[str, Any]] | None = None) -> None:
        """把 Mind 的输出交给渠道发送（先过 output 事件，让 filter 等插件净化）。

        targets 非空时按目标列表扇出（同一句话发到多个群/好友）；
        每条目标都会各自过一遍可发布白名单校验。
        """
        out = result.output
        if out is None or not out.is_visible():
            return

        attempted = False
        sent_any = False
        for item in self._fanout(out, targets):
            # 主动发言二次校验白名单（D7）：即便 Mind 产出了输出，也不越白名单乱发
            if proactive and self.policy is not None:
                target_type = "group" if item.target == "group_msg" else "user"
                target_id = item.group_id if item.target == "group_msg" else item.user
                if not await self.policy.can_publish(item.channel, target_type, target_id):
                    logger.info("目标 %s/%s 不在可发布白名单内，跳过本次主动发送",
                                target_type, target_id)
                    continue

            # 信号类事件：只通知订阅者（filter 会在此原地清洗 content）
            await self.bus.emit(TOPIC_OUTPUT, item)

            channel = self.channels.get(item.channel) or self.channels.get(self._primary)
            if channel is None:
                logger.warning("没有可用渠道，丢弃输出（%s）", item.channel)
                continue
            attempted = True
            if await channel.send(item):
                sent_any = True

        if sent_any:
            self._send_fails = 0
            if proactive:
                self._proactive_ts.append(time.time())
        elif attempted:
            self._send_fails += 1
            if self._send_fails >= self.send_fail_limit:
                self._circuit_open = True
                logger.warning("发送连续失败 %d 次，熔断主动发送", self._send_fails)