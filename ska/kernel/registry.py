"""注册表：工具注册表与渠道注册表。

可拓展性的关键机制（见方案「加功能不改内核」）：
    新增工具 = 写一个 ska/tools/xxx.py，用 @tool("名字") 装饰类；
    新增渠道 = 写一个 ska/io/xxx.py，用 @channel("名字") 装饰类；
    然后在 config.yaml 的 tools / channels 列表里加名字即可 —— 内核零改动。

本模块只提供「名字 → 对象」的通用容器，不依赖任何业务类型，
因此可以安全地留在 kernel 层。
"""
from __future__ import annotations

import logging
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Registry(Generic[T]):
    """通用注册表：把一个字符串名字映射到一个对象。"""

    def __init__(self, kind: str) -> None:
        self._kind = kind          # "tool" / "channel"，仅用于日志与报错
        self._items: dict[str, T] = {}

    def register(self, name: str, obj: T) -> None:
        """注册一个具名对象；同名重复注册会被拒绝（避免静默覆盖）。"""
        if name in self._items:
            logger.warning("%s 重复注册：%s（后注册的已忽略）", self._kind, name)
            return
        self._items[name] = obj

    def get(self, name: str) -> T | None:
        """按名取对象；不存在返回 None。"""
        return self._items.get(name)

    def require(self, name: str) -> T:
        """按名取对象；不存在则抛 KeyError（配置写错时尽早暴露）。"""
        if name not in self._items:
            raise KeyError(f"未知的{self._kind}：{name}（可用：{sorted(self._items)}）")
        return self._items[name]

    def names(self) -> list[str]:
        """已注册名字（排序后，便于稳定输出）。"""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __len__(self) -> int:
        return len(self._items)


# 全局单例注册表：装饰器直接写入这两个容器
TOOL_REGISTRY: Registry[Any] = Registry("tool")
CHANNEL_REGISTRY: Registry[Any] = Registry("channel")