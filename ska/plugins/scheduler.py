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
from ..kernel.event import TOPIC_WAKE, Event
from ..kernel.packet import CoreInput

logger = logging.getLogger(__name__)


class _Scheduler:
    """定时任务调度器。"""

    def __init__(self, bus: EventBus, tasks: list[dict[str, Any]]) -> None:
        self.bus = bus
        self.tasks = tasks
        self._task: asyncio.Task | None = None
        self._stop = False

    def start(self) -> None:
        if not self.tasks:
            logger.info("scheduler 插件：没有启用的任务，跳过")
            return
        self._task = asyncio.create_task(self._loop())
        logger.info("scheduler 插件已装载：%d 个任务", len(self.tasks))

    async def _loop(self) -> None:
        while not self._stop:
            now = datetime.now()
            upcoming = [(nxt, t) for t in self.tasks
                        if (nxt := _next_fire(t, now)) is not None]
            if not upcoming:
                logger.warning("scheduler：所有任务都无法解析出下次触发时间，退出")
                return
            nxt, task = min(upcoming, key=lambda pair: pair[0])
            delay = (nxt - now).total_seconds()
            if delay > 0:
                await asyncio.sleep(delay)
            if self._stop:
                break
            await self._fire(task, nxt)

    async def _fire(self, task: dict[str, Any], when: datetime) -> None:
        """投递一次 wake 事件。"""
        name = str(task.get("name", "task"))
        payload = str(task.get("payload", ""))
        logger.info("scheduler 触发任务：%s（%s）", name, when.strftime("%Y-%m-%d %H:%M"))
        await self.bus.publish(Event(
            topic=TOPIC_WAKE,
            payload=CoreInput(source="heartbeat", task=name, event="scheduler",
                              content=payload or f"定时任务 {name} 触发"),
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


def install(bus: EventBus, cfg) -> _Scheduler:
    """装载调度器（只取 enabled 的任务）。"""
    tasks = [t for t in cfg.scheduler.get("tasks", []) if t.get("enabled")]
    sched = _Scheduler(bus, tasks)
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