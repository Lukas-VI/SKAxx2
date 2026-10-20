"""上下文工程包（阶段三，见方案 §3）。

一句话定位：**SKA 决定「要多少上下文」，后端决定「怎么省显存」。**
本包负责前半句——把「人设 + 记忆召回 + 会话摘要 + 最近轮次」按预算组装成
一份可解释的上下文计划（ContextPlan），并把「占用了多少 token」回传出去，
供面板展示，也供后续 llama.cpp 适配器（动态 KV cache）使用。

对外只暴露三个东西：
    estimate_tokens / ContextBudget / ContextBuilder(+ContextPlan)
"""
from .budget import ContextBudget
from .builder import ContextBuilder, ContextPlan
from .estimator import estimate_tokens

__all__ = ["ContextBudget", "ContextBuilder", "ContextPlan", "estimate_tokens"]
