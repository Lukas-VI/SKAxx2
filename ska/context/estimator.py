"""token 估算（阶段三，见方案 §3.2）。

为什么不直接调 tokenizer：
    本地模型（llama.cpp / Ollama）的 tokenizer 各不相同，装一个统一分词器要么
    引依赖要么拉模型，成本远大于收益。方案明确「不新增依赖」，所以用**字符加权
    估算**：中日韩文字按 ~1.6 字符/token，ASCII 按 ~4 字符/token。

精度取舍：用于「预算控制 / 面板展示」足够（误差 ±20% 以内即可），
真要做「显存精确规划」时由后端 tokenizer 兜底 —— 这正是本包与适配器的分工。
"""
from __future__ import annotations

import math
from typing import Any, Iterable

# 估算系数：多少个「字符」算一个 token（越小越保守，即估出的 token 更多）
CJK_PER_TOKEN = 1.6      # 中文/日文/韩文：一个字大致 1 个多 token
ASCII_PER_TOKEN = 4.0    # 英文/数字/符号：平均 4 个字符一个 token

# 按此顺序判断是否为「CJK 字符」（含中文标点与全角符号区间）
_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3000, 0x303F),    # CJK 标点（、。「」等）
    (0x3040, 0x30FF),    # 平假名 / 片假名
    (0x3400, 0x4DBF),    # CJK 扩展 A
    (0x4E00, 0x9FFF),    # CJK 基本区（常用汉字）
    (0xAC00, 0xD7AF),    # 韩文音节
    (0xF900, 0xFAFF),    # CJK 兼容表意
    (0xFF00, 0xFFEF),    # 全角字符
)


def _is_cjk(ch: str) -> bool:
    """单个字符是否属于 CJK 区间。"""
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


def estimate_tokens(text: Any, *, ratio: float = 0.0) -> int:
    """估算一段文本的 token 数。

    Args:
        text:  任意可 str() 的内容（None → 空串）。
        ratio: 字符/token 的整体覆盖系数（来自 provider.token_ratio）。
               大于 0 时忽略自动判定，统一按 len(text)/ratio 计算。

    Returns:
        估算 token 数，至少 1（空文本给 0 —— 由调用方自行判断是否计入）。
    """
    content = "" if text is None else str(text)
    if not content:
        return 0
    if ratio and ratio > 0:
        return max(int(math.ceil(len(content) / float(ratio))), 1)
    cjk = sum(1 for ch in content if _is_cjk(ch))
    other = len(content) - cjk
    tokens = cjk / CJK_PER_TOKEN + other / ASCII_PER_TOKEN
    # 每条消息有 role 等固定开销，粗算 1~2 token；这里统一取整数上界
    return max(int(math.ceil(tokens)), 1)


def estimate_messages(messages: Iterable[Any]) -> int:
    """估算一组 chat 消息的总 token 数。

    兼容两种元素：dict{"role","content"} 与带 .content 属性的对象（如 Message/ChatMessage）。
    """
    total = 0
    for msg in messages or ():
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", "")
        total += estimate_tokens(content)
    return total


def truncate(text: str, max_tokens: int, *, ratio: float = 0.0) -> tuple[str, bool]:
    """按 token 上限截断文本，返回 (截断后文本, 是否发生截断)。

    从尾部截断（保留头部）：记忆块/摘要都是「开头更概括」的写法，
    砍尾巴比砍头更不容易丢掉主旨。
    """
    if max_tokens <= 0:
        return "", bool(text)
    if estimate_tokens(text, ratio=ratio) <= max_tokens:
        return text, False
    # 按字符逐步收缩：先用估算比例算出目标字符数，再线性夹逼
    if ratio and ratio > 0:
        target_chars = max(int(max_tokens * ratio), 1)
    else:
        target_chars = max(int(max_tokens * CJK_PER_TOKEN), 1)
    kept = text[:target_chars]
    # 夹逼到真正不超预算（最多收敛 8 轮，避免病态长文本上的循环开销）
    for _ in range(8):
        if estimate_tokens(kept, ratio=ratio) <= max_tokens or len(kept) <= 8:
            break
        kept = kept[: int(len(kept) * 0.8)]
    return kept.rstrip(), True
