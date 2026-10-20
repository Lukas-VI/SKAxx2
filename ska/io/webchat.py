"""webchat 渠道 —— WebUI「在线聊天」的收发两端（阶段三，见方案 §2.1）。

它为什么做成「渠道」而不是「后台功能模块」：
    渠道契约（见 io/base.py）已经白送了三样东西 —— 记忆落库、等级权限、
    心跳主动发言投递。只要把在线聊天接成渠道，这些能力不需要重写一遍；
    反之若在 web 层自己拼推理链，就会绕开内核的权限闸与 MindLine 追踪。

与 cli/qq 的差异（本渠道不直接碰网络）：
    * 收：不由本渠道监听。Web 线程经 ChatHub.submit() 组装 CoreInput 投总线；
    * 发：send() 把 CoreOutput 交给 ChatHub，前端 1 秒轮询取走。

依赖：ChannelDeps.extras["chat_hub"]（main.py 装配时注入）。
"""
from __future__ import annotations

import logging

from ..kernel.packet import CoreOutput
from .base import Channel, channel

logger = logging.getLogger(__name__)


@channel("webchat")
class WebChatChannel(Channel):
    """在线聊天渠道（纯内存搬运，无网络监听）。"""

    def __init__(self, deps) -> None:
        super().__init__(deps)
        # ChatHub 由 main.py 通过 extras 注入；缺失时渠道退化为「只丢日志」，
        # 保证没有后台组件时核心也能正常启动（与阶段二风格一致）。
        self.hub = deps.extras.get("chat_hub")
        self._started = False

    def _cfg(self) -> dict:
        """读 webchat 小节（优先 RuntimeSettings，支持后台热改）。"""
        return self.section()

    async def start(self) -> None:
        """本渠道无需建立接收循环：事件由后台 API 主动灌入。

        只做一次状态广播，让运行状态页能看到渠道是否到位。
        """
        cfg = self._cfg()
        self._started = True
        if not bool(cfg.get("enabled", True)):
            logger.info("webchat 渠道未启用（设置页可开启）")
            return
        if self.hub is None:
            logger.warning("webchat 渠道缺少 ChatHub（extras 未注入），在线聊天不可用")
            return
        # 把渠道的默认配置同步给 ChatHub：身份 / 默认落库开关（决策 E1、E3）
        self.hub.default_identity = str(cfg.get("identity") or cfg.get("user") or "owner")
        self.hub.default_persist = bool(cfg.get("persist", True))
        logger.info("webchat 渠道就绪（默认身份 %s，默认 persist=%s）",
                    self.hub.default_identity, self.hub.default_persist)

    async def send(self, out: CoreOutput) -> bool:
        """把模型回复投回对应会话队列（由 ChatHub 按 out.session 分派）。"""
        if self.hub is None:
            logger.warning("webchat 渠道未就绪，输出被丢弃")
            return False
        return self.hub.push(out.session, out)

    async def stop(self) -> None:
        """无后台任务需要取消（幂等）。"""
        self._started = False

    @property
    def connected(self) -> bool:
        """在线聊天的「在线」取决于后台是否就绪（供运行状态页展示）。"""
        return self.hub is not None
