"""上下文组装器（阶段三，见方案 §3.2）。

ContextBuilder.build() 把四路材料装成一份**可解释**的上下文计划：

    人设 system（固定）
      长期记忆召回（按配额截断）
      会话摘要 1~2 条（按配额截断）
      最近若干轮（从尾部回填，装到预算用尽）

产出的 ContextPlan 是「上下文意图」的唯一真相：面板读它、压缩策略读它、
将来的 llama.cpp 适配器也读它（SKA 声明要多少上下文，后端决定怎么省显存）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .budget import ContextBudget
from .estimator import estimate_messages, estimate_tokens, truncate

logger = logging.getLogger(__name__)


@dataclass
class ContextPlan:
    """一次上下文组装的完整产物（可解释、可回传）。"""

    messages: list[dict[str, Any]] = field(default_factory=list)   # 给执行者用的消息序列
    history: list[dict[str, Any]] = field(default_factory=list)    # 保留下来的最近轮次
    layers: dict[str, int] = field(default_factory=dict)           # 每层实际占用 token
    used_tokens: int = 0                                           # 总占用
    usable: int = 0                                                # 可用预算
    total: int = 0                                                 # 模型窗口
    dropped: int = 0                                               # 因预算挤掉的历史条数
    truncated: bool = False                                        # 记忆/摘要是否被截断
    need_compress: bool = False                                    # 是否达到压缩阈值

    # ------------------------------------------------------------------ 展示
    @property
    def ratio(self) -> float:
        """占用率（0~1，供状态条与 MindLine 显示）。"""
        return (self.used_tokens / self.usable) if self.usable else 0.0

    def render_history(self) -> str:
        """把保留的最近轮次渲染成给提示词用的对话文本。

        格式与阶段二的 mind.format_history 保持一致（对方：/你：），
        这样把 ContextBuilder 接进来后提示词结构不变，模型行为不会漂移。
        """
        lines: list[str] = []
        for msg in self.history:
            role, content = msg.get("role"), msg.get("content", "")
            if role == "user":
                lines.append(f"对方：{content}")
            elif role == "assistant":
                lines.append(f"你：{content}")
            else:
                lines.append(f"（{role}）{content}")
        return "\n".join(lines) if lines else "（还没有对话）"

    def as_dict(self) -> dict[str, Any]:
        """转成 JSON 友好的 dict（面板 / API / 轨迹 detail 用）。"""
        return {
            "used_tokens": self.used_tokens,
            "usable": self.usable,
            "total": self.total,
            "ratio": round(self.ratio, 4),
            "layers": dict(self.layers),
            "dropped": self.dropped,
            "truncated": self.truncated,
            "need_compress": self.need_compress,
        }


class ContextBuilder:
    """上下文组装器。

    Args:
        window_fn:    () -> 模型上下文窗口（token）；做成回调是为了「切换提供商后
                      立刻用新窗口」，而不是构造时读死。
        ratio_fn:     () -> 估算系数覆盖（provider.token_ratio，0 表示自动判定）。
        reserve_out:  给输出预留的 token。
        safety:       安全余量比例。
        compress_at:  占用率达到该比例即请求压缩（默认 0.8）。
        history_limit: 参与候选的最近轮次上限（防止一次性喂几千条进估算）。
        memory_frac / summary_frac: 记忆层与摘要层的配额占比。
    """

    def __init__(
        self,
        *,
        window_fn: Callable[[], int] | None = None,
        ratio_fn: Callable[[], float] | None = None,
        reserve_out: int = 1024,
        safety: float = 0.05,
        compress_at: float = 0.8,
        history_limit: int = 12,
        memory_frac: float = 0.25,
        summary_frac: float = 0.15,
    ) -> None:
        self._window_fn = window_fn or (lambda: 8192)
        self._ratio_fn = ratio_fn or (lambda: 0.0)
        self.reserve_out = max(int(reserve_out), 0)
        self.safety = float(safety)
        self.compress_at = min(max(float(compress_at), 0.1), 1.0)
        self.history_limit = max(int(history_limit), 1)
        self.memory_frac = float(memory_frac)
        self.summary_frac = float(summary_frac)

    # ------------------------------------------------------------------ 预算
    def budget(self) -> ContextBudget:
        """构造本次预算（窗口与系数即时读取，跟随提供商切换）。"""
        try:
            total = int(self._window_fn())
        except Exception:  # noqa: BLE001 —— 配置异常不该打断对话
            logger.warning("读取 context_window 失败，退回 8192", exc_info=True)
            total = 8192
        try:
            ratio = float(self._ratio_fn())
        except (TypeError, ValueError):
            ratio = 0.0
        return ContextBudget(total=total, reserve_out=self.reserve_out,
                             safety=self.safety, memory_frac=self.memory_frac,
                             summary_frac=self.summary_frac)

    def ratio(self) -> float:
        try:
            return float(self._ratio_fn())
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------ 组装
    def build(
        self,
        *,
        system: str = "",
        memory_block: str = "",
        summaries: Iterable[str] = (),
        history: Iterable[Any] = (),
        extra: Iterable[dict[str, Any]] = (),
    ) -> ContextPlan:
        """按预算组装上下文计划。

        Args:
            system:       人设 system prompt（固定层，不计入可压缩层）
            memory_block: 记忆召回块（画像 + 长期记忆 + 摘要的成品文本）
            summaries:    会话摘要列表（memory_block 为空时才有意义）
            history:      最近轮次（Message 对象或 {"role","content"} dict，任意顺序）
            extra:        追加消息（如工具观察结果），排在历史之后
        """
        ratio = self.ratio()
        budget = self.budget()
        plan = ContextPlan(usable=budget.usable, total=budget.total)

        # ---- ① 固定层：人设
        system_tokens = estimate_tokens(system, ratio=ratio)
        plan.layers["system"] = system_tokens

        # ---- ② 记忆召回层（超配额就截断，宁可少两条旧事）
        mem_text = memory_block or ""
        mem_quota = budget.memory_quota()
        if mem_text:
            mem_text, cut = truncate(mem_text, mem_quota, ratio=ratio)
            plan.truncated = plan.truncated or cut
        mem_tokens = estimate_tokens(mem_text, ratio=ratio)
        plan.layers["memory"] = mem_tokens

        # ---- ③ 会话摘要层（多条摘要合并后统一截断）
        summary_items = [s for s in (summaries or ()) if s]
        summary_text = "\n".join(f"- {s}" for s in summary_items)
        sum_quota = budget.summary_quota()
        if summary_text:
            summary_text, cut = truncate(summary_text, sum_quota, ratio=ratio)
            plan.truncated = plan.truncated or cut
        plan.layers["summary"] = estimate_tokens(summary_text, ratio=ratio)

        # ---- ④ 最近轮次层：从尾部回填（对话连续性命中率最高）
        items = [self._as_message(m) for m in list(history or ())][-self.history_limit:]
        fixed = system_tokens + mem_tokens + plan.layers["summary"]
        quota = budget.history_quota(fixed=fixed)
        kept: list[dict[str, Any]] = []
        used_history = 0
        for msg in reversed(items):
            cost = estimate_tokens(msg.get("content", ""), ratio=ratio)
            if used_history + cost > quota:
                break                                   # 装不下就停（不回填更早的）
            used_history += cost
            kept.append(msg)
        kept.reverse()                                  # 恢复时间顺序
        plan.dropped = len(items) - len(kept)
        plan.history = kept
        plan.layers["history"] = used_history

        # ---- ⑤ 追加层（工具结果等），同样受剩余预算约束
        used_extra = 0
        extra_msgs: list[dict[str, Any]] = []
        for msg in (extra or ()):
            cost = estimate_tokens(msg.get("content", ""), ratio=ratio)
            if used_history + used_extra + cost > quota:
                break
            used_extra += cost
            extra_msgs.append(dict(msg))
        if extra_msgs:
            plan.layers["extra"] = used_extra

        # ---- 组装总 token 与给执行者用的消息序列
        plan.used_tokens = (system_tokens + mem_tokens + plan.layers["summary"]
                            + used_history + used_extra)
        plan.messages = list(kept) + extra_msgs
        if mem_text:
            # 执行者视角：记忆块作为一条 system 消息前置（与阶段二 _agent_context 同款）
            plan.messages = [{"role": "system", "content": f"【你想起的旧事】\n{mem_text}"}] \
                             + plan.messages

        # ---- 判定是否需要压缩（供 Mind 决定要不要滚动摘要 + 记忆整理）
        plan.need_compress = bool(plan.usable and plan.ratio >= self.compress_at)
        return plan

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _as_message(item: Any) -> dict[str, str]:
        """把 Message 对象/dict 统一成 {"role","content"}。"""
        if isinstance(item, dict):
            return {"role": str(item.get("role") or "user"),
                    "content": str(item.get("content") or "")}
        return {"role": str(getattr(item, "role", "user")),
                "content": str(getattr(item, "content", ""))}

    def occupancy_of(self, messages: Iterable[Any]) -> int:
        """估算一组消息的 token（供用量页「平均 prompt token」等展示）。"""
        return estimate_messages(messages)
