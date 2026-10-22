"""事件总线 EventBus。

这是对老项目最关键的修复之一：
    老 L2Bot_server.recive_data 是「单槽位」，处理一条消息期间到达的新消息
    会被直接覆盖 → 丢消息。
    这里改为 asyncio.Queue 无界队列：来多少收多少，主循环用 drain() 一次取空，
    再批量处理，从根本上消除覆盖丢失。

同时统一了「唤醒」语义：publish 入队时置位 asyncio.Event，
主循环 wait(timeout) 要么被唤醒（有事件），要么超时（= 一次心跳）。
这正是老 core.py heart_beat() 的 `wait_for(event.wait(), timeout=60/bpm)` 的升级版。
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Awaitable, Callable

from .event import Event

logger = logging.getLogger(__name__)

# 订阅者回调签名：接收一条 Event，可以是协程函数
Handler = Callable[[Event], Awaitable[None] | None]


class EventBus:
    """极简发布/订阅 + 队列唤醒总线。

    用法：
        bus.subscribe("output", handler)      # 注册订阅者（插件用）
        await bus.publish(Event("input", pkt))  # 投递输入事件（入队+唤醒）
        await bus.emit("output", out)          # 只通知订阅者（不进主循环）
        events = await bus.drain()             # 主循环取空队列
        fired = await bus.wait(timeout)        # 有事件=True，超时=False
    """

    def __init__(self) -> None:
        # 主循环的输入队列：无界，保证不丢消息
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        # 有事件时置位，供 wait() 立即返回；避免忙轮询
        self._wake = asyncio.Event()
        # 主题 → 订阅者列表
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    # ------------------------------------------------------------------ 订阅
    def subscribe(self, topic: str, handler: Handler) -> None:
        """订阅某主题。可多次调用注册多个处理者，按注册顺序依次调用。"""
        self._handlers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> bool:
        """取消订阅（插件热卸载/热重载时使用）；返回是否真的移除了。

        为什么需要它：插件卸载时若不摘掉订阅，旧实例会继续收到事件——
        热重载后就会出现「同一个事件被新旧两份插件各处理一次」的幽灵重复。
        按对象身份移除，且容错（重复取消不抛异常）。
        """
        handlers = self._handlers.get(topic)
        if not handlers:
            return False
        for i, existing in enumerate(handlers):
            if existing is handler:
                del handlers[i]
                return True
        return False

    # ------------------------------------------------------------------ 投递
    async def publish(self, event: Event) -> None:
        """投递「输入类」事件：先通知该主题订阅者，再入队并唤醒主循环。

        注意事件处理顺序：订阅者通常做的是「归一化/去重/审计」等轻量预处理，
        真正需要 LLM 推理的处理留给主循环 drain 后统一做。
        """
        await self._notify(event)
        self._queue.put_nowait(event)
        self._wake.set()

    async def emit(self, topic: str, payload=None) -> None:
        """发送「信号类」事件：只通知订阅者，不进入主循环。

        用于 output（filter 拦截）、heartbeat（统计）等不需要再推理的事件，
        避免主循环把自己的输出又当成输入，造成隐式自循环。
        """
        await self._notify(Event(topic=topic, payload=payload))

    async def _notify(self, event: Event) -> None:
        """调用某主题的所有订阅者。单个订阅者异常不影响其他订阅者。"""
        for handler in list(self._handlers.get(event.topic, ())):
            try:
                result = handler(event)
                # 兼容同步/异步两种回调
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001 —— 总线必须容错，插件崩了不能拖垮内核
                logger.exception("事件订阅者异常 topic=%s handler=%s",
                                 event.topic, getattr(handler, "__qualname__", handler))

    # ------------------------------------------------------------------ 消费
    async def drain(self) -> list[Event]:
        """取空队列，返回全部待处理事件。主循环每次唤醒调用一次。"""
        events: list[Event] = []
        while not self._queue.empty():
            try:
                events.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:  # 理论上不会发生，防御性处理
                break
        return events

    async def wait(self, timeout: float) -> bool:
        """等待事件唤醒；超时返回 False（= 一次心跳）。

        返回 True 后立即清除唤醒标志：若无新事件，下一次 wait 会正常阻塞，
        不会因为残留标志而空转。
        """
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return False
        self._wake.clear()
        return True

    # ------------------------------------------------------------------ 辅助
    def qsize(self) -> int:
        """当前待处理事件数（观测用）。"""
        return self._queue.qsize()

    def wakeup(self) -> None:
        """不投递事件、仅唤醒等待中的 wait()（用于优雅退出等场景）。"""
        self._wake.set()