"""渠道基类与注册装饰器。

Channel 的契约（务必与 main.py / beat.py 保持一致）：
    start()            启动接收（内部把收到的消息转成 CoreInput 并 publish 到总线）
    send(CoreOutput)   发送一条输出，返回 bool 表示是否成功（供熔断统计）
    stop()             优雅停止（取消后台任务）

渠道只负责「搬运」，不做任何决策；所有推理在 loop 层。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..kernel.bus import EventBus
from ..kernel.packet import CoreInput, CoreOutput
from ..kernel.registry import CHANNEL_REGISTRY


@dataclass
class ChannelDeps:
    """注入给渠道的共享依赖。"""

    bus: EventBus
    cfg: Any                                    # kernel.config.Config
    settings: Any = None                        # admin.settings.RuntimeSettings（可为 None）
    extras: dict[str, Any] = field(default_factory=dict)   # 渠道需要的额外依赖


class Channel(ABC):
    """所有渠道的基类。"""

    name: str = ""                 # 由 @channel("名字") 填充

    def __init__(self, deps: ChannelDeps) -> None:
        self.deps = deps
        self.bus = deps.bus

    def section(self) -> dict[str, Any]:
        """读取本渠道的配置小节。

        优先走 RuntimeSettings（config.yaml ⊕ 后台覆盖层，支持热改配置）；
        未注入时退回直接读 config.yaml（向后兼容阶段一的构造方式）。
        """
        if self.deps.settings is not None:
            return self.deps.settings.section(self.name)
        return self.deps.cfg.section(self.name)

    @abstractmethod
    async def start(self) -> None:
        """启动接收任务（非阻塞：内部 create_task）。"""
        raise NotImplementedError

    @abstractmethod
    async def send(self, out: CoreOutput) -> bool:
        """发送一条输出；返回是否成功。"""
        raise NotImplementedError

    async def stop(self) -> None:
        """停止渠道（默认无操作，需要清理资源的渠道覆写）。"""
        return None

    # 便捷方法：把一条输入投递到总线（入队 + 唤醒主循环）
    async def publish_input(self, packet: CoreInput) -> None:
        from ..kernel.event import TOPIC_INPUT, Event
        await self.bus.publish(Event(topic=TOPIC_INPUT, payload=packet))


def channel(name: str):
    """类装饰器：把渠道类注册进全局注册表。"""

    def decorator(cls: type[Channel]) -> type[Channel]:
        cls.name = name
        CHANNEL_REGISTRY.register(name, cls)
        return cls

    return decorator


def build_channels(names: list[str], deps: ChannelDeps) -> dict[str, Channel]:
    """按配置名单实例化渠道。"""
    built: dict[str, Channel] = {}
    for name in names:
        cls = CHANNEL_REGISTRY.require(name)
        built[name] = cls(deps)
    return built