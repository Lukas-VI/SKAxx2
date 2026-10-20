"""结构化输出解析与重试。

老项目的痛点：模型手写 JSON，常带 ``` 包裹或多余解释文字，`json.loads` 一失败就
静默降级，链路不可控。这里给出确定性流程：

    优先 response_format=json_object
      → 若输出被 max_tokens 截断（finish_reason=length / 中转 structured_output_failed）
         则按 2 倍预算自动重试（见 TRUNCATE_BUMP_FACTOR），而不是直接判失败
      → 剥离 code fence / 截取最外层 JSON 对象
      → json.loads
      → 失败则带「修复指令」重试 1 次
      → 仍失败抛 StructuredError，由调用方降级为「纯文本直发」
        （明确不回退到老 prompt.json 那种「再推理」死循环）
"""
from __future__ import annotations

import json
import logging
from typing import Any, Sequence

from .gateway import LLMTruncatedError
from .models import ChatMessage

logger = logging.getLogger(__name__)

# 输出被截断（finish_reason=length / 中转 structured_output_failed）时的自动抬升策略：
#   每遇到一次截断，就把 max_tokens 乘以下面的倍数再试；到达上限仍截断才放弃。
# 上限 8192 是「够大的结构化输出」与「不至于把成本/延迟拉爆」之间的折中。
TRUNCATE_BUMP_FACTOR = 2
MAX_STRUCTURED_TOKENS = 8192

# 匹配 ```json ... ``` 或 ``` ... ```
_FENCE_RE = None


class StructuredError(RuntimeError):
    """结构化输出无法解析（经重试后仍失败）。"""


def _fence_pattern():
    # 延迟导入 re 并缓存，避免模块级开销
    global _FENCE_RE
    if _FENCE_RE is None:
        import re
        _FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
    return _FENCE_RE


def strip_code_fence(text: str) -> str:
    """剥离 markdown code fence；没有 fence 时原样返回（去除首尾空白）。"""
    text = (text or "").strip()
    match = _fence_pattern().search(text)
    return match.group(1).strip() if match else text


def extract_json_object(text: str) -> str | None:
    """从字符串中截取「最外层平衡的 JSON 对象」。

    逐字符扫描并跟踪花括号深度，同时正确跳过字符串内部的 { } 与转义字符。
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None  # 花括号不闭合


def decode_json(text: str) -> Any:
    """把模型输出解码成 Python 对象；失败抛 StructuredError。"""
    cleaned = strip_code_fence(text)
    # 第一尝试：整体就是 JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # 第二尝试：截取最外层对象
    candidate = extract_json_object(cleaned)
    if candidate is not None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            raise StructuredError(f"JSON 解析失败：{e}；原文片段：{candidate[:120]}") from e
    raise StructuredError(f"未找到 JSON 对象；原文片段：{cleaned[:120]}")


async def structured_chat(
    gateway: Any,
    messages: Sequence[ChatMessage],
    *,
    retry: int = 1,
    temperature: float | None = None,
    task: str = "",
) -> dict[str, Any]:
    """发起一次「要求 JSON 对象」的调用，解析失败自动重试。

    gateway 采用鸭子类型：只要有 `async chat(messages, *, json_mode=..., temperature=...)`
    即可（测试里的 FakeLLM 因此无需继承 LLMGateway）。

    Args:
        retry: 解析失败后的额外尝试次数（方案要求 1 次）。
        task:  用量统计用的任务标签；为空时不传该参数，
               以兼容不接受 task 的网关替身（如测试里的 FakeLLM）。

    Returns:
        解析后的 dict。

    Raises:
        StructuredError: 重试后仍无法得到合法 JSON 对象。
    """
    convo: list[ChatMessage] = list(messages)
    last_err: Exception | None = None
    budget: int | None = None      # None=用网关默认 max_tokens；被截断后逐级抬高

    for attempt in range(retry + 1):
        # 仅当给了 task 标签才透传该参数（保持对 FakeLLM 等替身的兼容）；
        # max_tokens 仅在「已发生截断」后才显式传入，避免影响未截断的常规路径。
        kwargs: dict[str, Any] = {"json_mode": True, "temperature": temperature}
        if task:
            kwargs["task"] = task
        if budget is not None:
            kwargs["max_tokens"] = budget
        try:
            resp = await gateway.chat(convo, **kwargs)
        except LLMTruncatedError as e:
            # 输出被截断 = 预算不足（而非调用失败）：抬高预算再试，而不是直接降级。
            last_err = e
            base = budget if budget is not None else _base_budget(gateway)
            raised = min(base * TRUNCATE_BUMP_FACTOR, MAX_STRUCTURED_TOKENS)
            if raised <= base:                     # 已到上限，无法再加 → 交给上层降级
                logger.warning("结构化输出仍被截断且已达预算上限 %d：%s", base, e)
                break
            budget = raised
            logger.warning("结构化输出被截断，抬高输出预算至 %d 后重试（原因：%s）", budget, e)
            continue
        try:
            data = decode_json(resp.content)
        except StructuredError as e:
            last_err = e
            logger.warning("结构化解析失败（第 %d 次）：%s", attempt + 1, e)
            if attempt < retry:
                # 附带模型原文 + 修复指令，做一次「就地纠错」重试
                convo = convo + [
                    ChatMessage(role="assistant", content=resp.content),
                    ChatMessage(
                        role="user",
                        content=("上一条输出不是合法 JSON。请只输出一个 JSON 对象，"
                                 "不要包含 markdown 代码块、注释或解释文字。"),
                    ),
                ]
                continue
            break
        if not isinstance(data, dict):
            last_err = StructuredError(f"期望 JSON 对象，得到 {type(data).__name__}")
            logger.warning("结构化输出类型不符：%s", last_err)
            break
        return data

    raise last_err or StructuredError("结构化输出失败")


def _base_budget(gateway: Any) -> int:
    """取网关的默认 max_tokens（鸭子类型；取不到时退回 1024）。"""
    try:
        return max(int(getattr(gateway, "max_tokens", 1024)), 1)
    except (TypeError, ValueError):
        return 1024