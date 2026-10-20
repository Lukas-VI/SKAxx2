"""上下文工程测试（阶段三 · 方案 §3）。

覆盖三块：
    * estimator —— 字符加权 token 估算与截断（不引 tokenizer）；
    * ContextBudget —— 总预算与记忆/摘要/历史三层配额；
    * ContextBuilder —— ContextPlan 组装、历史回填、压缩阈值判定。

全部为纯计算，不需要事件循环，故用同步 TestCase。
"""
from __future__ import annotations

import unittest

from ska.context import ContextBudget, ContextBuilder, estimate_tokens
from ska.context.estimator import estimate_messages, truncate


# =========================================================================== 估算
class TestEstimator(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens(None), 0)
        self.assertEqual(estimate_messages([]), 0)

    def test_cjk_costlier_than_ascii(self):
        """同样字符数，中文估出的 token 更多（CJK 1.6 字符/token ＜ ASCII 4）。"""
        self.assertGreater(estimate_tokens("中" * 100), estimate_tokens("a" * 100))
        self.assertGreaterEqual(estimate_tokens("你好世界"), 1)

    def test_ratio_override(self):
        """给定 ratio 时统一按 len/ratio 计算，忽略自动判定。"""
        self.assertEqual(estimate_tokens("abcdefgh", ratio=4.0), 2)

    def test_estimate_messages_accepts_dict_and_object(self):
        class _Msg:
            def __init__(self, content):
                self.content = content

        total = estimate_messages([{"role": "user", "content": "你好"}, _Msg("hello")])
        self.assertGreater(total, 0)

    def test_truncate_keeps_head(self):
        text = "中" * 1000
        out, cut = truncate(text, 10)
        self.assertTrue(cut)
        self.assertLess(len(out), len(text))
        short, cut2 = truncate("短", 10)
        self.assertFalse(cut2)
        self.assertEqual(short, "短")


# =========================================================================== 预算
class TestBudget(unittest.TestCase):
    def test_usable_and_quotas_are_consistent(self):
        b = ContextBudget(total=8192, reserve_out=1024, safety=0.05)
        # 可用 = (窗口 − 输出预留) × (1 − 安全余量)
        self.assertEqual(b.usable, int((8192 - 1024) * 0.95))
        self.assertGreater(b.memory_quota(), 0)
        self.assertGreater(b.summary_quota(), 0)
        self.assertEqual(b.history_quota(fixed=0),
                         b.usable - b.memory_quota() - b.summary_quota())

    def test_small_window_has_floor_protection(self):
        b = ContextBudget(total=10, reserve_out=0, safety=0.0)
        self.assertGreaterEqual(b.usable, 128)          # 下限保护，避免除零/负预算
        self.assertGreaterEqual(b.memory_quota(), 64)
        self.assertGreaterEqual(b.summary_quota(), 32)


# =========================================================================== 组装
class TestContextBuilder(unittest.TestCase):
    def _builder(self, **kw) -> ContextBuilder:
        kw.setdefault("window_fn", lambda: 8192)
        kw.setdefault("ratio_fn", lambda: 0.0)
        return ContextBuilder(**kw)

    def test_plan_layers_and_messages(self):
        b = self._builder()
        hist = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好呀"}]
        plan = b.build(system="你是助手", memory_block="他喜欢巧克力", history=hist)

        self.assertEqual(plan.total, 8192)
        self.assertEqual(len(plan.history), 2)          # 窗口够大，全保留
        self.assertEqual(plan.dropped, 0)
        self.assertGreater(plan.used_tokens, 0)
        self.assertIn("history", plan.layers)
        # 记忆块以 system 消息前置（执行者视角）
        self.assertTrue(plan.messages[0]["content"].startswith("【你想起的旧事】"))
        self.assertIn("巧克力", plan.messages[0]["content"])
        # as_dict 供面板/轨迹使用，ratio 落在 0~1
        d = plan.as_dict()
        self.assertIn("used_tokens", d)
        self.assertGreaterEqual(plan.ratio, 0.0)

    def test_render_history_format(self):
        b = self._builder()
        plan = b.build(history=[{"role": "user", "content": "甲"},
                                {"role": "assistant", "content": "乙"}])
        text = plan.render_history()
        self.assertIn("对方：甲", text)
        self.assertIn("你：乙", text)

    def test_history_dropped_when_over_budget(self):
        """极小窗口 → 只能从尾部留最近几条，dropped > 0。"""
        b = self._builder(window_fn=lambda: 1024, reserve_out=800)
        hist = [{"role": "user", "content": "很" * 100} for _ in range(20)]
        plan = b.build(system="", history=hist)
        self.assertGreater(plan.dropped, 0)
        self.assertLess(len(plan.history), 20)

    def test_need_compress_when_over_threshold(self):
        b = self._builder(window_fn=lambda: 2048, reserve_out=0, compress_at=0.3)
        plan = b.build(history=[{"role": "user", "content": "中" * 1000}])
        self.assertTrue(plan.need_compress)             # 占用率 ≥ 0.3

    def test_extra_messages_appended_after_history(self):
        b = self._builder()
        plan = b.build(history=[{"role": "user", "content": "hi"}],
                       extra=[{"role": "tool", "content": "观测结果"}])
        self.assertEqual(plan.messages[-1]["role"], "tool")

    def test_window_fn_failure_falls_back(self):
        """窗口读取异常不该打断组装（退回 8192）。"""
        def bad():
            raise RuntimeError("配置炸了")

        b = ContextBuilder(window_fn=bad, ratio_fn=lambda: 0.0)
        plan = b.build(history=[{"role": "user", "content": "hi"}])
        self.assertEqual(plan.total, 8192)


if __name__ == "__main__":
    unittest.main()
