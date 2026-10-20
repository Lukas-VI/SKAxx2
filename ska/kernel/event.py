"""事件数据类与主题常量。

迁移自老项目 events/event.py 的「轮询 + asyncio.Event 通知」模型：
老的 Event.slot() 每秒轮询一次检查状态变化，效率低且有延迟；
新模型改为「谁产生事件谁 publish」，由 EventBus 统一入队与唤醒（见 bus.py）。

主题（topic）在此集中定义，避免各处硬编码字符串拼错。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 主题常量
# ---------------------------------------------------------------------------
# 以下主题会被心跳主循环 drain() 取出并进入一轮推理（「输入类」事件）：
TOPIC_INPUT = "input"                  # 通用用户输入（渠道归一化后）
TOPIC_WAKE = "wake"                    # 主动唤醒（scheduler 定时、外部触发）
TOPIC_CHANNEL_STATUS = "channel_status"  # 渠道状态变化（断线/重连）
TOPIC_SYSTEM = "system"                # 系统事件（如启动、关闭）

# 以下主题只通知订阅者，不进入主循环（「信号类」事件）：
TOPIC_OUTPUT = "output"                # 待发送的 CoreOutput（filter 插件在此拦截）
TOPIC_HEARTBEAT = "heartbeat"          # 每记一次心跳（观测/统计用）
TOPIC_TRACE = "trace"                  # MindLine 轨迹事件（阶段三；只落库/广播，不推理）

# 主循环关心的输入类主题集合
INPUT_TOPICS = frozenset({TOPIC_INPUT, TOPIC_WAKE, TOPIC_CHANNEL_STATUS, TOPIC_SYSTEM})


@dataclass(slots=True)
class Event:
    """一条总线事件。

    Attributes:
        topic:   主题字符串，取值见上方常量。决定由谁处理。
        payload: 载荷。输入类事件通常是 CoreInput；输出类通常是 CoreOutput；
                 定时唤醒通常是 {"task": 名称, "text": 提示}。
        ts:      事件产生时间戳（秒）。默认取当前时间。
    """

    topic: str
    payload: Any = None
    ts: float = field(default_factory=time.time)

    def is_input(self) -> bool:
        """是否为「输入类」事件（需要进入主循环推理）。"""
        return self.topic in INPUT_TOPICS