"""标准数据包与内部数据契约（pydantic 强类型）。

迁移自老项目 core/IO_Package.py 的 CoreInput / CoreOutput，但做两处关键改良：

  1. 「来源」由构造方**显式声明**，不再像老代码那样按字段名去「猜」来源
     （老做法遇到字段缺失/混装就会误判，且 type/dict 双形态极易出错）。
  2. 全量换成 pydantic 模型：字段类型、必填性、枚举取值都在运行时校验，
     非法结构第一时间报错，而不是静默降级。
"""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field

# ===========================================================================
# 一、对外收发数据包
# ===========================================================================

# 输入来源枚举（显式声明）
InputSource = Literal[
    "cli_message",    # 本地 CLI 输入（默认开发渠道）
    "qq_message",     # QQ 消息（迁移自老 QQHttpServer）
    "self_response",  # 机器人自己的输出回灌（保留老 self_response 语义）
    "system_event",   # 系统事件（启动/关闭/渠道断线重连）
    "heartbeat",      # 心跳主动唤醒
]


class CoreInput(BaseModel):
    """进入系统的一条输入。

    与老 CoreInput 的字段对应关系：
        old.pack/type/pack_source → 现在由 source 显式给出；
        old.user/content/time/type/card → 保留同名/近义字段。
    """

    source: InputSource                       # 显式来源，不再靠猜
    content: str = ""                         # 正文（用户说了什么 / 事件描述）
    user: str = ""                            # 用户标识（QQ user_id 或 CLI 名）
    user_name: str = ""                       # 昵称（老 card）
    channel: str = "cli"                      # 来自哪个渠道
    message_type: str = "private"             # private / group
    group_id: str = ""                        # 群号（群消息时）
    event: str = ""                           # 系统事件名（source=system_event 时）
    task: str = ""                            # 定时任务名（source=heartbeat 时）
    ts: float = Field(default_factory=time.time)
    raw: dict[str, Any] = Field(default_factory=dict)  # 原始报文，便于排查

    def is_user_message(self) -> bool:
        """是否为真实用户消息（决定是否放行、记忆是否落库）。"""
        return self.source in ("cli_message", "qq_message")


# 输出目标枚举
OutputTarget = Literal[
    "group_msg",     # 发到群
    "private_msg",   # 发到私聊
    "internal",      # 只进内部记忆，对话可见性为「不可见用户」（继承老 /skip 语义）
]


class CoreOutput(BaseModel):
    """系统要发出的一条输出。"""

    content: str                              # 正文
    target: OutputTarget = "private_msg"      # 发送目标
    channel: str = "cli"                      # 走哪个渠道
    user: str = ""                            # 私聊对象
    group_id: str = ""                        # 群号
    ts: float = Field(default_factory=time.time)

    def is_visible(self) -> bool:
        """是否需要真正发给用户（internal 不发送，只落记忆）。"""
        return self.target != "internal"


# ===========================================================================
# 二、内部推理契约
# ===========================================================================


class ToolCall(BaseModel):
    """一次工具调用意图（由 LLM 结构化输出解析而来）。"""

    name: str                                  # 工具名（对应 TOOL_REGISTRY）
    args: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    """AgentLoop 的一步轨迹（think → act → observe）。

    说明：轨迹用于回填上下文与可观测日志；不展示给用户。
    """

    step: int                                  # 第几步（从 1 开始）
    thought: str = ""                          # 该步的思考
    tool: str = ""                             # 调用的工具名（空=没调工具）
    args: dict[str, Any] = Field(default_factory=dict)
    observation: str = ""                      # 工具返回（已截断）


class AgentResult(BaseModel):
    """AgentLoop 执行结果（执行者视角）。"""

    final: str = ""                            # 最终可对外的文本
    steps: list[AgentStep] = Field(default_factory=list)
    stopped: bool = False                      # 是否因达到 max_steps 被截断
    error: str = ""                            # 执行期错误摘要（可为空）


class Decision(BaseModel):
    """Mind 的决策結果（决策者视角）。"""

    action: Literal["speak", "silence", "remember", "tool"]
    content: str = ""                          # action=speak 时要说的话
    tool: ToolCall | None = None               # action=tool 时的调用意图
    memory: str = ""                           # action=remember 时写入的记忆文本
    importance: float = 0.5                    # 重要性 0~1（记忆排序用）
    reason: str = ""                           # 决策理由（只进日志/内独白）


class MindResult(BaseModel):
    """Mind 一轮唤醒的完整产物。"""

    decision: Decision
    output: CoreOutput | None = None           # action=speak 时组装好的输出
    agent: AgentResult | None = None           # 执行工具时的执行者结果
    monologue: str = ""                        # 内独白（只进日志/记忆，不进用户输出）
    reflection: str = ""                       # 反思（可选的二次修订说明）