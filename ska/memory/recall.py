"""记忆检索组装。

一次唤醒开始前（Mind 步骤①），把三类信息拼成一段「你想起的事」文本：
    1. 用户画像（精确匹配，不耗 LLM）
    2. FTS5 trigram 命中的长期记忆（按 importance·exp(-Δt/30d) 排序 top_k）
    3. 最近 2 条会话摘要
拼好后作为额外上下文塞进后续的决策/执行提示词。
"""
from __future__ import annotations

from typing import Any

from .store import DEFAULT_USER


async def build_memory_block(store: Any, query: str, user: str = "",
                             cfg: dict[str, Any] | None = None) -> str:
    """组装记忆块文本；没有任何命中时返回空串。"""
    cfg = cfg or {}
    top_k = int(cfg.get("recall_top_k", 5))
    half_life = float(cfg.get("recall_half_life_days", 30))
    blocks: list[str] = []

    # 1) 用户画像
    profile = await store.get_profile(user or DEFAULT_USER)
    nick = profile.get("nick") or ""
    facts = profile.get("facts") or []
    if nick or facts:
        lines = []
        if nick:
            lines.append(f"- 对方希望被称为「{nick}」")
        lines.extend(f"- {f}" for f in facts)
        blocks.append("关于对方的画像：\n" + "\n".join(lines))

    # 2) 长期记忆检索
    if query.strip():
        hits = await store.search_long(query, top_k=top_k, half_life_days=half_life,
                                       user=user)
        if hits:
            blocks.append("你想起的事：\n" + "\n".join(f"- {h}" for h in hits))

    # 3) 最近会话摘要
    sessions = await store.recent_sessions(2, user)
    if sessions:
        blocks.append("最近的对话摘要：\n" + "\n".join(f"- {s}" for s in sessions))

    return "\n\n".join(blocks)