"""SessionRegistry —— 多会话的工作记忆登记表（阶段三，见方案 §2.2）。

背景：阶段一/二的 SessionContext 是**单例**（main.py 构造一次，BeatLoop 持有）。
阶段三要支持「在线聊天多开」，就必须让每个会话各有各的窗口。

做法：
    键 key = f"{channel}:{user}:{session_id}"
    值 SessionContext（窗口大小沿用 memory.working_window）

向后兼容（本文件最重要的一条约定）：
    旧渠道（CLI / QQ）不传 session → session_id 取默认值 "default"，
    于是它们的 key 恒定不变，永远命中同一个 SessionContext —— 行为与阶段二完全一致。

多开语义（决策 E1）：
    长期记忆 / 画像 / 权限按**身份（user）**共享；只有「对话窗口 + 是否落库」
    按 session 隔离。所以同一身份开两个会话，记忆是通的，窗口是分开的。
"""
from __future__ import annotations

from typing import Any

from ..kernel.context import SessionContext

# 缺省段位：渠道 / 用户 / 会话缺失时的占位（保证 key 结构稳定）
DEFAULT_CHANNEL = "cli"
DEFAULT_USER = "default"
DEFAULT_SESSION = "default"


class SessionRegistry:
    """key → SessionContext 的登记表。所有方法都是同步的（内存字典，够快）。"""

    def __init__(self, window: int = 20) -> None:
        self.window = max(int(window), 1)                 # 单会话窗口条数
        self._contexts: dict[str, SessionContext] = {}    # key → 上下文

    # ------------------------------------------------------------------ 键
    @staticmethod
    def key_for(payload: Any) -> str:
        """从输入包算出会话键。

        优先读 CoreInput 的 channel/user/session 三个字段；任何对象（含测试替身）
        缺字段都不报错，用 getattr 兜底取默认值——即「老渠道不传就是默认会话」。
        """
        channel = str(getattr(payload, "channel", "") or DEFAULT_CHANNEL)
        user = str(getattr(payload, "user", "") or DEFAULT_USER)
        session = str(getattr(payload, "session", "") or DEFAULT_SESSION)
        return f"{channel}:{user}:{session}"

    # ------------------------------------------------------------------ 取用
    def context_for(self, payload: Any) -> SessionContext:
        """取（不存在则创建）某输入对应的会话上下文。"""
        key = self.key_for(payload)
        ctx = self._contexts.get(key)
        if ctx is None:
            ctx = SessionContext(self.window)
            self._contexts[key] = ctx
        return ctx

    def get(self, key: str) -> SessionContext | None:
        return self._contexts.get(key)

    def ensure(self, key: str) -> SessionContext:
        """按「现成 key」取会话（webchat 关闭会话后清理用）。"""
        ctx = self._contexts.get(key)
        if ctx is None:
            ctx = SessionContext(self.window)
            self._contexts[key] = ctx
        return ctx

    def drop(self, key: str) -> bool:
        """移除某个会话的窗口（关闭在线会话时调用，内存随之释放）。"""
        return self._contexts.pop(key, None) is not None

    # ------------------------------------------------------------------ 观测
    def keys(self) -> list[str]:
        """当前存活的会话键（后台展示/调试用）。"""
        return list(self._contexts)

    def stats(self) -> dict[str, Any]:
        """会话数 + 各会话的窗口占用（供运行状态页展示）。"""
        return {
            "count": len(self._contexts),
            "window": self.window,
            "sessions": {k: len(v) for k, v in self._contexts.items()},
        }
