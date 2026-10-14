"""CLI 渠道：默认渠道，便于在没有 QQ 环境下开发与验收。

读取键盘的注意点（踩过的坑）：
    直接用 asyncio.to_thread(input) 会让阻塞的 input() 跑在 ThreadPoolExecutor 的
    非守护线程上；退出时 concurrent.futures 会在解释器关闭阶段 join 它 → 进程假死，
    必须再按一次回车才能结束。因此这里改为「自建守护线程读 stdin + 把行投回事件循环」，
    既不阻塞事件循环，也不拖住退出。
"""
from __future__ import annotations

import asyncio
import logging
import threading

from ..kernel.event import TOPIC_SYSTEM, Event
from ..kernel.packet import CoreInput, CoreOutput
from .base import Channel, channel

logger = logging.getLogger(__name__)

# 退出指令（也可以直接 Ctrl+C）
_QUIT_WORDS = {"/quit", "/exit", "/退出"}


@channel("cli")
class CLIChannel(Channel):
    """终端交互渠道。"""

    def __init__(self, deps) -> None:
        super().__init__(deps)
        self._lines: asyncio.Queue[str | None] = asyncio.Queue()
        self._stopped = False
        self._consumer: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._prompt = str(self.section().get("prompt", "你> "))

    async def start(self) -> None:
        print("=" * 60)
        print("SKAxx2 已启动（CLI 渠道）。直接输入说话；输入 /quit 或 Ctrl+C 退出。")
        print("=" * 60)
        # 守护线程负责阻塞读取；读到一行就投回事件循环队列
        self._loop = asyncio.get_running_loop()
        threading.Thread(target=self._read_stdin, name="cli-stdin", daemon=True).start()
        self._consumer = asyncio.create_task(self._consume())

    def _read_stdin(self) -> None:
        """（守护线程）阻塞读取标准输入，把结果投回事件循环。"""
        loop = self._loop
        while not self._stopped:
            try:
                line = input(self._prompt)
            except (EOFError, KeyboardInterrupt):
                line = None            # None 作为「输入结束」哨兵
            except Exception:          # noqa: BLE001 —— 输入流异常直接结束
                line = None
            loop.call_soon_threadsafe(self._lines.put_nowait, line)
            if line is None:
                return

    async def _consume(self) -> None:
        """（事件循环）消费输入行并投递事件。"""
        while not self._stopped:
            line = await self._lines.get()
            if line is None:
                # 输入结束（EOF）：请求退出
                await self._request_shutdown("输入已关闭")
                return
            text = line.strip()
            if not text:
                continue
            if text in _QUIT_WORDS:
                await self._request_shutdown("用户请求退出")
                return
            await self.publish_input(CoreInput(
                source="cli_message",
                content=text,
                user="local",
                user_name="你",
                channel=self.name,
                message_type="private",
            ))

    async def _request_shutdown(self, reason: str) -> None:
        """走「系统事件」路径请求 shutdown，由心跳循环统一收尾。"""
        await self.bus.publish(Event(
            topic=TOPIC_SYSTEM,
            payload=CoreInput(source="system_event", event="shutdown",
                              content=reason, channel=self.name),
        ))

    async def send(self, out: CoreOutput) -> bool:
        """把输出打印到终端。"""
        print(f"\n[赛里斯库妮娅] {out.content}\n")
        return True

    async def stop(self) -> None:
        self._stopped = True
        if self._consumer and not self._consumer.done():
            self._consumer.cancel()
            try:
                await self._consumer
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass