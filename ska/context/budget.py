"""上下文预算与分层配额（阶段三，见方案 §3.2）。

总预算 = 模型上下文窗口 − 预留输出 − 安全余量。
分层配额（方案里列的 5 层，这里归成 4 个「可压缩层」）：

    system / 人设       固定，不可压（要塞得下，否则后面一切免谈）
    长期记忆召回        按份额（默认 25%）
    会话摘要            按份额（默认 15%）
    最近轮次            剩余全部 —— 从窗口尾部回填，能装多少装多少

为什么「最近轮次」不设死配额：对话连续性最值钱，宁可少召回两条旧事，
也要保证最近几轮完整（这也正是老项目「无限 append 撑爆窗口」的修正方向）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextBudget:
    """一次上下文组装的预算分配。

    Attributes:
        total:       模型上下文窗口（token），来自 provider.context_window
        reserve_out: 给本次输出预留的 token（模型还要生成回复）
        safety:      安全余量比例（0.05 = 再留 5%，容忍估算误差）
        memory_frac: 记忆召回层占比
        summary_frac:会话摘要层占比
    """

    total: int = 8192
    reserve_out: int = 1024
    safety: float = 0.05
    memory_frac: float = 0.25
    summary_frac: float = 0.15

    # ------------------------------------------------------------------ 计算
    @property
    def usable(self) -> int:
        """可用的「输入」预算（已扣掉输出预留与安全余量）。"""
        total = max(int(self.total), 256)
        raw = total - max(int(self.reserve_out), 0)
        return max(int(raw * (1.0 - max(min(self.safety, 0.5), 0.0))), 128)

    def memory_quota(self) -> int:
        return max(int(self.usable * self.memory_frac), 64)

    def summary_quota(self) -> int:
        return max(int(self.usable * self.summary_frac), 32)

    def history_quota(self, fixed: int = 0) -> int:
        """最近轮次的配额 = 可用预算 − 固定部分 − 记忆 − 摘要（不足则给 0）。"""
        left = self.usable - max(int(fixed), 0) - self.memory_quota() - self.summary_quota()
        return max(left, 0)

    def as_dict(self) -> dict[str, int | float]:
        """给面板看的配额快照。"""
        return {
            "total": int(self.total),
            "reserve_out": int(self.reserve_out),
            "safety": float(self.safety),
            "usable": self.usable,
            "memory_quota": self.memory_quota(),
            "summary_quota": self.summary_quota(),
        }
