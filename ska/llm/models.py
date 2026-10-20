"""LLM 消息与响应模型。

说明：ToolCall 是内核数据契约（packet.Decision.tool 依赖它），
本模块直接复用 kernel.packet.ToolCall 并在此再导出，保证全项目只有一种 ToolCall 类型。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..kernel.packet import ToolCall  # 复用内核契约，再导出（唯一 ToolCall 类型）

__all__ = ["ChatMessage", "LLMResponse", "ToolCall"]


class ChatMessage(BaseModel):
    """一条 chat 消息（OpenAI 兼容格式）。"""

    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None           # 工具消息对应的工具名（可选）
    tool_call_id: str | None = None   # 工具消息回填时的调用 id（可选）

    def to_payload(self) -> dict[str, Any]:
        """转成发给网关的 dict，自动省略空的可选字段。"""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload


class LLMResponse(BaseModel):
    """网关一次调用的返回。

    Attributes:
        content:   文本内容（纯文本或 JSON 字符串）。
        tool_call: 若模型要求调用工具，这里是解析后的第一个调用意图。
        model:     实际使用的模型名（回显）。
        usage:     token 用量（若服务端返回）。
        raw:       原始 message 对象，便于排查。
        finish_reason: 结束原因（stop/length/tool_calls…）。为 "length" 表示输出被
                       max_tokens 截断——上层据此「自动加预算重试」（见 structured_chat）。
    """

    content: str = ""
    tool_call: ToolCall | None = None
    model: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str = ""                   # 可选：结束原因（截断检测用）

    def has_tool_call(self) -> bool:
        return self.tool_call is not None