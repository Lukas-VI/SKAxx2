"""定时唤醒插件（迁移自老 events/e_Scheduler.py）。

变化点：
    老实现：每分钟轮询一次（Task 里 sleep 60 再比对时间），触发精度差、空转多。
    新实现：装载时按 cron/at 表达式**算出下一次触发时刻**，只 sleep 到那一点；
            触发后投递一个 wake 事件，由心跳循环按 task 名路由给 Mind。

支持写法（够用即可）：
    at:   "HH:MM"          每天固定时刻
    cron: "M H * * *"      分 时 日 月 周；分/时支持 *、n、a,b、*/n；
                           日/月/周非 * 时会被忽略（只记一条警告）。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any

from ..kernel.bus import EventBus
from ..kernel.config import load_yaml
from ..kernel.event import TOPIC_WAKE, Event
from ..kernel.packet import CoreInput

logger = logging.getLogger(__name__)

# 睡眠分段：最多睡这么久就醒来重算一次。
# 目的：改完 config/scheduler.yaml 不必重启进程，最迟 30 秒后自动生效。
REFRESH_SECONDS = 30.0

# 定时任务表文件（相对项目根目录）
SCHEDULER_FILE = "config/scheduler.yaml"

# 插件设置项：后台「插件拓展」页读这份 schema 自动生成表单。
# 约定见 ska/plugins/manager.py::settings_schema。
SETTINGS: dict[str, Any] = {
    "title": "定时任务",
    "fields": [{
        "key": "targets",
        "label": "消息通道",
        "type": "targets",
        "multiple": True,
        "default": [],
        "help": ("日程触发后把消息发到这些目标（可多选）。留空时默认发到 QQ 的默认群"
                 "（config.yaml 的 qq.default_group）；没有默认群则沿用最近一次对话的来路。"),
    }],
}


def _parse_targets(values: Any) -> list[dict[str, Any]]:
    """把设置里的 "qq|group|123456" 转成路由字典（格式不对的条目直接忽略）。"""
    out: list[dict[str, Any]] = []
    for item in values or []:
        parts = str(item or "").split("|")
        if len(parts) != 3:
            continue
        channel, target_type, target_id = (p.strip() for p in parts)
        if not channel or not target_id or target_type not in ("group", "user"):
            continue
        out.append({"channel": channel, "target_type": target_type,
                    "target_id": target_id})
    return out


def _default_targets(cfg: Any) -> list[dict[str, Any]]:
    """默认消息通道 = 群聊：取 QQ 小节的 default_group（没配就返回空）。"""
    try:
        group = str((cfg.section("qq") or {}).get("default_group") or "").strip()
    except (AttributeError, TypeError):
        group = ""
    return [{"channel": "qq", "target_type": "group", "target_id": group}] if group else []


def _read_tasks(cfg: Any) -> list[dict[str, Any]]:
    """从磁盘重读任务表（只保留 enabled）。

    为什么不直接用 cfg.scheduler：Config 只在进程启动时读一次 YAML，
    之后改文件不会反映到内存里——表现就是「改了时间/打开了开关，到点却没动静」。
    因此每轮调度都重新读盘；读失败（文件被编辑器占用、YAML 写坏）时退回启动快照，
    保证调度器不会因为一次误编辑就整体罢工。
    """
    try:
        data = load_yaml(SCHEDULER_FILE)
        raw = data.get("tasks")
        if isinstance(raw, list):
            return [t for t in raw if isinstance(t, dict) and t.get("enabled")]
    except Exception:  # noqa: BLE001 —— 配置读坏时不能拖垮调度线程
        logger.warning("读取 %s 失败，沿用启动时的任务表", SCHEDULER_FILE, exc_info=True)
    raw = getattr(cfg, "scheduler", {}) or {}
    return [t for t in raw.get("tasks", [])
            if isinstance(t, dict) and t.get("enabled")]


class _Scheduler:
    """定时任务调度器。"""

    def __init__(self, bus: EventBus, tasks: list[dict[str, Any]],
                 targets: list[dict[str, Any]] | None = None,
                 cfg: Any = None) -> None:
        self.bus = bus
        self.tasks = tasks
        self.targets = list(targets or [])     # 日程触发后的投递目标（可多个）
        self.cfg = cfg                         # 用于每轮重读任务表（可为 None）
        self._task: asyncio.Task | None = None
        self._stop = False

    def start(self) -> None:
        if not self.tasks:
            # 不再直接放弃：任务可能只是暂时全关，等用户改完配置就能被捡起来
            logger.info("scheduler 插件：当前没有启用的任务，等待配置变更（每 %.0fs 重读）",
                        REFRESH_SECONDS)
        self._task = asyncio.create_task(self._loop())
        logger.info("scheduler 插件已装载：%d 个任务", len(self.tasks))

    async def _loop(self) -> None:
        while not self._stop:
            # 每轮从磁盘重读任务表：编辑 scheduler.yaml 后无需重启进程
            if self.cfg is not None:
                self.tasks = _read_tasks(self.cfg)
            if not self.tasks:
                await asyncio.sleep(REFRESH_SECONDS)
                continue
            now = datetime.now()
            upcoming = [(nxt, t) for t in self.tasks
                        if (nxt := _next_fire(t, now)) is not None]
            if not upcoming:
                logger.warning("scheduler：所有任务都无法解析出下次触发时间，等待配置修正")
                await asyncio.sleep(REFRESH_SECONDS)
                continue
            nxt, task = min(upcoming, key=lambda pair: pair[0])
            delay = (nxt - now).total_seconds()
            # 分段睡眠：离触发还早时先睡一小段，醒来重新比较任务表，
            # 这样「把时间改到更早」也能在 REFRESH_SECONDS 内被感知。
            if delay > REFRESH_SECONDS:
                await asyncio.sleep(REFRESH_SECONDS)
                continue
            if delay > 0:
                await asyncio.sleep(delay)
            if self._stop:
                break
            await self._fire(task, nxt)

    async def _fire(self, task: dict[str, Any], when: datetime) -> None:
        """投递一次 wake 事件（带上配置好的消息通道）。"""
        name = str(task.get("name", "task"))
        payload = str(task.get("payload", ""))
        logger.info("scheduler 触发任务：%s（%s），投递目标 %d 个",
                    name, when.strftime("%Y-%m-%d %H:%M"), len(self.targets))
        # 触发来路取第一个目标：Mind 据此生成提示词/默认路由；
        # 完整目标列表放进 raw["_targets"]，由心跳在发送阶段扇出到每个通道。
        route: dict[str, Any] = {}
        if self.targets:
            first = self.targets[0]
            if str(first.get("target_type")) == "group":
                route = {"channel": first.get("channel", ""),
                         "message_type": "group",
                         "group_id": first.get("target_id", "")}
            else:
                route = {"channel": first.get("channel", ""),
                         "message_type": "private",
                         "user": first.get("target_id", "")}
        await self.bus.publish(Event(
            topic=TOPIC_WAKE,
            payload=CoreInput(source="heartbeat", task=name, event="scheduler",
                              content=payload or f"定时任务 {name} 触发",
                              raw={"_targets": list(self.targets)}, **route),
        ))
        # 同一分钟内避免重复触发
        await asyncio.sleep(1)

    async def stop(self) -> None:
        self._stop = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


def install(bus: EventBus, cfg, settings=None) -> _Scheduler:
    """装载调度器。

    settings 是 RuntimeSettings（由 PluginManager 按签名注入），用于读取后台
    配置的「消息通道」；没有后台设置时退回 QQ 默认群（群聊）。

    任务表从磁盘读取（而不是 cfg.scheduler 快照），并在运行期持续重读，
    所以改 config/scheduler.yaml 后不需要重启进程。
    """
    tasks = _read_tasks(cfg)
    values: dict[str, Any] = {}
    if settings is not None:
        try:
            values = settings.section("plugin:scheduler")
        except (AttributeError, TypeError):
            values = {}
    targets = _parse_targets(values.get("targets")) or _default_targets(cfg)
    sched = _Scheduler(bus, tasks, targets, cfg=cfg)
    sched.start()
    return sched


# --------------------------------------------------------------------------- 时间计算
def _next_fire(task: dict[str, Any], now: datetime) -> datetime | None:
    """算出该任务在 now 之后的下一次触发时刻。"""
    if task.get("at"):
        return _next_from_at(str(task["at"]), now)
    if task.get("cron"):
        return _next_from_cron(str(task["cron"]), now)
    return None


def _next_from_at(spec: str, now: datetime) -> datetime | None:
    """"HH:MM" → 下一次该时刻（今天已过则顺延到明天）。"""
    try:
        hour, minute = (int(x) for x in spec.split(":", 1))
    except ValueError:
        logger.warning("无法解析 at 表达式：%s", spec)
        return None
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def _next_from_cron(spec: str, now: datetime) -> datetime | None:
    """极简 cron：只认真调度「分」「时」，其余字段为 * 时忽略。"""
    parts = spec.split()
    if len(parts) < 2:
        logger.warning("无法解析 cron 表达式：%s", spec)
        return None
    if any(p != "*" for p in parts[2:6]):
        logger.warning("cron 的 日/月/周 字段暂不支持，将被忽略：%s", spec)
    minutes = _expand(parts[0], 0, 59)
    hours = _expand(parts[1], 0, 23)
    if not minutes or not hours:
        return None
    # 逐天、逐时、逐分找最早的可触发时刻（最多往后看一年）
    day = now.date()
    for _ in range(370):
        for hour in sorted(hours):
            for minute in sorted(minutes):
                candidate = datetime.combine(day, datetime.min.time()).replace(
                    hour=hour, minute=minute)
                if candidate > now:
                    return candidate
        day += timedelta(days=1)
    return None


def _expand(field: str, lo: int, hi: int) -> list[int]:
    """展开 cron 字段：* / n / a,b / */n。"""
    values: set[int] = set()
    for token in field.split(","):
        token = token.strip()
        if token == "*":
            values.update(range(lo, hi + 1))
        elif token.startswith("*/"):
            step = int(token[2:] or "1")
            values.update(range(lo, hi + 1, max(step, 1)))
        elif token.isdigit():
            value = int(token)
            if lo <= value <= hi:
                values.add(value)
    return sorted(values)