"""会话上下文 SessionContext —— 工作记忆（定窗）。

修复老项目的硬伤之一：`Core.messages` 只 append 不裁剪，长会话下上下文无限膨胀，
最终撑爆模型窗口。这里用 deque(maxlen=...) 定窗，超出自动丢弃最旧消息。

注意：deque 丢弃的只是「送给 LLM 的短期窗口」；原始流水是否落库由 memory 层决定，
本类不负责持久化。
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class Message:
    """一条对话消息。"""

    role: str        # "system" / "user" / "assistant" / "tool"
    content: str
    ts: float = 0.0

    def to_chat(self) -> dict[str, str]:
        """转成 OpenAI 兼容的 chat 消息格式。"""
        return {"role": self.role, "content": self.content}


class SessionContext:
    """当前会话的工作记忆，固定窗口大小。"""

    def __init__(self, window: int = 20) -> None:
        # maxlen 生效：追加超过 window 条时，最旧的自动淘汰
        self._window = window
        self._msgs: deque[Message] = deque(maxlen=window)
        self.started_ts = time.time()   # 会话开始时间（用于会话边界判定）
        self.last_ts = self.started_ts  # 最后一条消息时间（空闲判定）

    # ------------------------------------------------------------------ 写入
    def add(self, role: str, content: str, ts: float | None = None) -> Message:
        """追加一条消息并更新时间戳。"""
        msg = Message(role=role, content=content, ts=ts or time.time())
        self._msgs.append(msg)
        self.last_ts = msg.ts
        return msg

    def clear(self) -> None:
        """清空窗口（开启新会话时调用）。"""
        self._msgs.clear()
        self.started_ts = time.time()
        self.last_ts = self.started_ts

    # ------------------------------------------------------------------ 读取
    def recent(self, n: int | None = None) -> list[Message]:
        """取最近 n 条（n 为 None 则取满窗口）。"""
        if n is None or n >= len(self._msgs):
            return list(self._msgs)
        return list(self._msgs)[-n:]

    def to_chat(self) -> list[dict[str, str]]:
        """把整个窗口转成 chat 消息列表，供 LLM 使用。"""
        return [m.to_chat() for m in self._msgs]

    def idle_seconds(self) -> float:
        """距最后一条消息的空闲秒数（会话边界判定用）。"""
        return time.time() - self.last_ts

    def __len__(self) -> int:
        return len(self._msgs)