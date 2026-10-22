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
    "cli_message",      # 本地 CLI 输入（默认开发渠道）
    "qq_message",       # QQ 消息（迁移自老 QQHttpServer）
    "webchat_message",  # WebUI「在线聊天」输入（阶段三：多开会话）
    "self_response",    # 机器人自己的输出回灌（保留老 self_response 语义）
    "system_event",     # 系统事件（启动/关闭/渠道断线重连）
    "heartbeat",        # 心跳主动唤醒
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
    # ---- 阶段三：多会话（见方案 §2.2） ----
    # 会话标识，用于把「同时开着的几个话题」隔离开。
    # 旧渠道（CLI/QQ）不传 → 空串统一落「默认会话」，行为与阶段二完全一致。
    session: str = ""
    # 沙盒标记：persist=false 的在线会话置 True —— 照常推理，但不写任何记忆库。
    sandbox: bool = False
    # ---- 图片 IO：随消息一起进来的图片/表情包段 ----
    # 每项形如 {"kind": "image|sticker|face|mface", "url": "...", "file": "...",
    #          "sub_type": "1", "summary": "..."}；由渠道解析 OneBot 消息段时填充。
    # 文本里仍保留「[图片]」占位，视觉解析结果由 Mind 追加到正文（见 _describe 链路）。
    images: list[dict[str, Any]] = Field(default_factory=list)
    ts: float = Field(default_factory=time.time)
    raw: dict[str, Any] = Field(default_factory=dict)  # 原始报文，便于排查

    def is_user_message(self) -> bool:
        """是否为真实用户消息（决定是否放行、记忆是否落库）。"""
        return self.source in ("cli_message", "qq_message", "webchat_message")


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
    # 随消息一起发出的图片：本地绝对路径或 http(s) URL。
    # QQ 渠道会把它转成 OneBot image 段（URL 原样传，本地文件转 base64://）。
    images: list[str] = Field(default_factory=list)
    # 阶段三：输出来自哪个会话。webchat 渠道据此把回复投回对应的会话队列；
    # 旧渠道不传 → 空串，行为不变。
    session: str = ""
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
    # 工具链里通过 image_send 附加的图片（本地路径/URL）；Mind 会并入 CoreOutput.images
    images: list[str] = Field(default_factory=list)
    stopped: bool = False                      # 是否因达到 max_steps 被截断
    # 截断原因（max_steps / repeat；空=正常结束）——上层据此给出可读反馈，
    # 不再出现「被截断但没人知道」的静默失败。
    stop_reason: str = ""
    error: str = ""                            # 执行期错误摘要（可为空）


class Route(BaseModel):
    """Agent 自主指定的「这条输出投递到哪」（阶段三增强）。

    设计取舍：老做法靠「记住上一条用户消息的来路」(_remember_route) 让主动发言
    回到刚聊过的会话——那是**被动**兜底。但本 Agent 是「并行单实例」的，
    同时开着多个会话时，她应当能**主动**决定这句话该发到哪个渠道/会话/群。
    因此让决策模型可选地给出 route：
        * route 为空（None）→ 沿用触发来路继承（既有行为，向后兼容）；
        * route 有值 → 优先按它决定 channel / user / session / message_type / group_id。
    空字段表示「该项沿用触发来路」，便于只覆盖其中一项。

    注：_remember_route 仍是心跳主动路径的兜底（心跳触发本身没有来路），
        两者并存、互不冲突。
    """

    channel: str = ""                          # cli / qq / webchat（空=沿用触发来路）
    user: str = ""                             # 目标用户 id（空=沿用）
    session: str = ""                          # 目标会话（空=沿用）
    message_type: str = ""                     # private / group（空=沿用）
    group_id: str = ""                         # 群号（空=沿用）


class Decision(BaseModel):
    """Mind 的决策結果（决策者视角）。"""

    action: Literal["speak", "silence", "remember", "tool", "sticker", "think"]
    content: str = ""                          # action=speak 时要说的话（sticker 时为配文/兜底）
    tool: ToolCall | None = None               # action=tool 时的调用意图
    memory: str = ""                           # action=remember 时写入的记忆文本
    # ---- 图片/表情包（阶段四） ----
    # sticker：表情包检索词（如「开心」「无语」）；本地库命中就直接发表情包，
    #          没命中则退化为 speak（用 content 兜底）。
    sticker: str = ""
    # images：speak 时可附带本地路径或 URL（如代码画图后的 PNG）。
    images: list[str] = Field(default_factory=list)
    importance: float = 0.5                    # 重要性 0~1（记忆排序用）
    reason: str = ""                           # 决策理由（只进日志/内独白）
    # ---- 阶段三增强：自主路由 + 自主休眠（见方案 §2） ----
    route: Route | None = None                 # 自主投递目标（None=沿用触发来路）
    # 自主休眠：Agent 在思考回环里可以决定「接下来 N 秒不主动发声」，
    # 覆盖默认心跳节奏（心跳仅作兜底）；期间任何输入/插件事件都会立即把它唤醒。
    sleep_seconds: float = 0.0


class MindResult(BaseModel):
    """Mind 一轮唤醒的完整产物。"""

    decision: Decision
    output: CoreOutput | None = None           # action=speak 时组装好的输出
    agent: AgentResult | None = None           # 执行工具时的执行者结果
    monologue: str = ""                        # 内独白（只进日志/记忆，不进用户输出）
    reflection: str = ""                       # 反思（可选的二次修订说明）


# ===========================================================================
# 三、可观测契约（阶段三 MindLine，见方案 §4.1）
# ===========================================================================
# 泳道（lane）：MindLine 页面自上而下的三行，一次交互的每个阶段落在其中一行。
TraceLane = Literal["input", "think", "act"]

# 事件类型（kind）：决定页面上的图标/配色与筛选维度。
TraceKind = Literal[
    "user_msg",      # 收到用户消息
    "wake",          # 心跳/系统事件唤醒
    "silence",       # 权限闸拒绝或决定沉默
    "monologue",     # 内独白
    "decision",      # 决策结果
    "degrade",       # 结构化失败降级为纯文本
    "tool_call",     # 工具调用（参数）
    "tool_result",   # 工具返回（结果摘要 + 耗时）
    "reflection",    # 反思（二次修订）
    "compress",      # 上下文压缩 / 记忆整理
    "memory_write",  # 写入长期记忆或画像
    "output",        # 最终输出投递给渠道
    "image",         # 来图解析（视觉模型读图/表情包入库）
    "sticker",       # 发表情包（本地库命中）
    "error",         # 异常
]


class TraceEvent(BaseModel):
    """MindLine 的一条轨迹事件。

    设计要点：
        * `trace_id` 把「同一次交互」的各阶段串成一列（页面按列排）；
        * `lane` 决定它落在三行中的哪一行；
        * `seq` 由 TraceLog 统一分配（全局单调递增），前端用 `after=seq` 做增量拉取；
        * 这是「信号类」事件（走 TOPIC_TRACE），不进入主循环推理。
    """

    trace_id: str = ""                        # 一次交互的唯一标识
    seq: int = 0                              # 全局递增序号（由 TraceLog 赋值）
    ts: float = Field(default_factory=time.time)
    session: str = ""                         # 所属会话（多开隔离）
    lane: TraceLane = "think"
    kind: TraceKind = "monologue"
    title: str = ""                           # 一行摘要（页面主展示）
    detail: str = ""                          # 完整细节（点开看）
    tokens: int = 0                           # 该步涉及的 token 量（上下文占用）
    sandbox: bool = False                     # 是否沙盒会话（不落库）