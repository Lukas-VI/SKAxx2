"""多提供商管理测试（阶段二新增）。

覆盖 ProviderManager：首次迁移、增删改查、切换 active 时网关热更新、
密钥不外泄（public_list 只回 has_key）。
"""
from __future__ import annotations

import os
import tempfile
import unittest

from ska.llm.providers import PRESETS, ProviderError, ProviderManager


class TestProviderManager(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "providers.yaml")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ------------------------------------------------------------------ 迁移
    def test_migrate_from_fallback(self):
        """没有 providers.yaml 时，用 config.yaml 的 llm 小节迁移出默认实例。"""
        mgr = ProviderManager(self.path, fallback={
            "base_url": "http://localhost:11434/v1", "api_key": "ollama",
            "model": "qwen2.5:7b", "temperature": 0.8, "timeout": 60})
        self.assertEqual(mgr.active_id(), "default")
        self.assertEqual(mgr.gateway.model, "qwen2.5:7b")
        self.assertTrue(os.path.exists(self.path))            # 已落盘

    def test_empty_without_fallback(self):
        mgr = ProviderManager(self.path)
        self.assertEqual(mgr.active_id(), "")
        self.assertEqual(mgr.list(), [])
        self.assertEqual(mgr.presets(), PRESETS)

    # ------------------------------------------------------------------ 增删改查
    def _manager(self) -> ProviderManager:
        return ProviderManager(self.path)

    def test_add_update_remove(self):
        mgr = self._manager()
        item = mgr.add({"id": "deepseek", "label": "DeepSeek",
                        "base_url": "https://api.deepseek.com/v1",
                        "api_key": "sk-x", "model": "deepseek-chat"})
        self.assertEqual(item["id"], "deepseek")
        self.assertEqual(mgr.active_id(), "deepseek")          # 首个实例自动激活

        mgr.update("deepseek", {"model": "deepseek-reasoner"})
        self.assertEqual(mgr.get("deepseek")["model"], "deepseek-reasoner")
        self.assertEqual(mgr.gateway.model, "deepseek-reasoner")   # active 改了 → 网关热更新

        mgr.remove("deepseek")
        self.assertEqual(mgr.list(), [])
        self.assertEqual(mgr.active_id(), "")

    def test_add_duplicate_id_raises(self):
        mgr = self._manager()
        mgr.add({"id": "p1", "base_url": "http://x", "model": "m"})
        with self.assertRaises(ProviderError):
            mgr.add({"id": "p1", "base_url": "http://y", "model": "m"})

    def test_validate_required_fields(self):
        mgr = self._manager()
        with self.assertRaises(ProviderError):
            mgr.add({"id": "bad", "base_url": "", "model": "m"})       # 缺 base_url
        with self.assertRaises(ProviderError):
            mgr.add({"id": "bad id!", "base_url": "http://x", "model": "m"})  # id 非法

    def test_auto_id(self):
        mgr = self._manager()
        item = mgr.add({"preset": "ollama", "base_url": "http://x", "model": "m"})
        self.assertEqual(item["id"], "ollama")
        second = mgr.add({"preset": "ollama", "base_url": "http://y", "model": "m"})
        self.assertEqual(second["id"], "ollama-2")             # 冲突自动加序号

    def test_activate_and_remove_active_falls_back(self):
        mgr = self._manager()
        mgr.add({"id": "p1", "base_url": "http://x", "model": "m1"})
        mgr.add({"id": "p2", "base_url": "http://y", "model": "m2"})
        mgr.activate("p2")
        self.assertEqual(mgr.active_id(), "p2")
        self.assertEqual(mgr.gateway.provider_id, "p2")
        self.assertEqual(mgr.gateway.model, "m2")

        mgr.remove("p2")                                       # 删掉 active → 自动切到剩余第一个
        self.assertEqual(mgr.active_id(), "p1")
        self.assertEqual(mgr.gateway.model, "m1")

    def test_update_missing_raises(self):
        with self.assertRaises(ProviderError):
            self._manager().update("nope", {"model": "x"})

    # ------------------------------------------------------------------ 展示
    def test_public_list_hides_key(self):
        mgr = self._manager()
        mgr.add({"id": "p1", "base_url": "http://x", "model": "m", "api_key": "secret"})
        mgr.add({"id": "p2", "base_url": "http://y", "model": "m"})
        public = {p["id"]: p for p in mgr.public_list()}
        self.assertTrue(public["p1"]["has_key"])
        self.assertFalse(public["p2"]["has_key"])
        self.assertTrue(public["p1"]["is_active"])
        self.assertNotIn("api_key", public["p1"])              # 不回明文

    def test_persist_and_reload(self):
        mgr = self._manager()
        mgr.add({"id": "p1", "base_url": "http://x", "model": "m1"})
        mgr.activate("p1")
        again = ProviderManager(self.path)                     # 重新读同一份 YAML
        self.assertEqual(again.active_id(), "p1")
        self.assertEqual(again.get("p1")["model"], "m1")

    def test_active_info(self):
        mgr = self._manager()
        mgr.add({"id": "p1", "base_url": "http://x/v1", "model": "m1"})
        info = mgr.active_info()
        self.assertEqual(info["id"], "p1")
        self.assertEqual(info["model"], "m1")

    # ------------------------------------------------------------------ 上下文窗口（阶段三 §3.1）
    def test_context_window_added_and_used(self):
        """新增时手填的 context_window 会被保存，并成为压缩窗口的来源。"""
        mgr = self._manager()
        mgr.add({"id": "p1", "base_url": "http://x/v1", "model": "m1",
                 "context_window": 32768})
        self.assertEqual(mgr.get("p1")["context_window"], 32768)
        self.assertEqual(mgr.context_window(), 32768)          # active=首个实例
        self.assertEqual(mgr.active_info()["context_window"], 32768)

    def test_context_window_defaults_to_8192(self):
        """未填/非法值统一退回默认 8192（绝不抛异常）。"""
        mgr = self._manager()
        mgr.add({"id": "p1", "base_url": "http://x", "model": "m1"})
        self.assertEqual(mgr.context_window(), 8192)
        mgr.update("p1", {"context_window": "abc"})            # 非法字符串 → 回落
        self.assertEqual(mgr.get("p1")["context_window"], 8192)

    def test_context_window_lower_bound(self):
        """过小的窗口被夹到下限 256（避免预算算出 0）。"""
        mgr = self._manager()
        mgr.add({"id": "p1", "base_url": "http://x", "model": "m1", "context_window": 10})
        self.assertEqual(mgr.context_window(), 256)

    def test_context_window_switch_provider(self):
        """切换 active 提供商后，压缩窗口即时跟随新提供商（回调式读取）。"""
        mgr = self._manager()
        mgr.add({"id": "p1", "base_url": "http://x", "model": "m1", "context_window": 8192})
        mgr.add({"id": "p2", "base_url": "http://y", "model": "m2", "context_window": 65536})
        self.assertEqual(mgr.context_window(), 8192)
        mgr.activate("p2")
        self.assertEqual(mgr.context_window(), 65536)

    def test_presets_carry_context_window(self):
        """预设模板应带上各模型的常见窗口值，供新增表单预填。"""
        windows = {p["id"]: p.get("context_window") for p in PRESETS}
        self.assertEqual(windows["deepseek"], 65536)
        self.assertEqual(windows["openai"], 128000)
        self.assertTrue(all(v and v >= 256 for v in windows.values()))


if __name__ == "__main__":
    unittest.main()