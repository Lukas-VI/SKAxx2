"""记忆检索组装（阶段三升级：三层记忆一起注入）。

一次唤醒开始前（Mind 步骤①），把以下信息拼成一段「你想起的事」文本：
    1. 核心记忆（core_facts）——始终注入，不依赖检索（MemGPT/Letta core memory 风格）
    2. 用户画像（profile，精确匹配，不耗 LLM）
    3. 结构化事实（entity_facts）——按 2-gram 相关度 × 重要度 × 衰减召回（Mem0 风格）
    4. 长期记忆（long_memory）——FTS5 bm25 × 重要度 × 时间衰减（三因子排序）
    5. 最近 2 条会话摘要

为什么分三层：核心事实要「永远在场」（身份、称呼、底线偏好）；结构化事实要「可去重、
能覆盖」（喜好的变化会被更新而不是并存矛盾）；长期记忆是「可检索的原文」。
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
    core_limit = int(cfg.get("core_limit", 8))
    facts_top_k = int(cfg.get("facts_top_k", 4))
    blocks: list[str] = []

    # 1) 核心记忆：始终注入（不检索），这是「她永远记得的事」
    core = await store.list_core(user=user, limit=core_limit)
    if core:
        blocks.append("你一直记着的事：\n" + "\n".join(f"- {c}" for c in core))

    # 2) 用户画像
    profile = await store.get_profile(user or DEFAULT_USER)
    nick = profile.get("nick") or ""
    facts = profile.get("facts") or []
    if nick or facts:
        lines = []
        if nick:
            lines.append(f"- 对方希望被称为「{nick}」")
        lines.extend(f"- {f}" for f in facts)
        blocks.append("关于对方的画像：\n" + "\n".join(lines))

    # 3) 结构化事实：相关度 × 重要度 × 衰减，去重后的「槽位式」记忆
    if query.strip():
        entity_hits = await store.search_facts(query, top_k=facts_top_k, user=user)
        if entity_hits:
            blocks.append("你记得的事实：\n" + "\n".join(f"- {h}" for h in entity_hits))

    # 4) 长期记忆检索（三因子排序）
    if query.strip():
        hits = await store.search_long(query, top_k=top_k, half_life_days=half_life,
                                       user=user)
        if hits:
            blocks.append("你想起的事：\n" + "\n".join(f"- {h}" for h in hits))

    # 5) 最近会话摘要
    sessions = await store.recent_sessions(2, user)
    if sessions:
        blocks.append("最近的对话摘要：\n" + "\n".join(f"- {s}" for s in sessions))

    return "\n\n".join(blocks)
