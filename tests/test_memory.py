"""记忆层测试：中文子串召回、画像 upsert、会话摘要、流水落库。

用临时目录里的 sqlite，不污染 data/ska.db。
"""
from __future__ import annotations

import os
import tempfile
import unittest

from ska.memory.store import MemoryStore


class TestMemoryStore(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(os.path.join(self._tmp.name, "test.db"))

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self._tmp.cleanup()

    # ------------------------------------------------------------------ 长期记忆
    async def test_chinese_substring_recall(self):
        """FTS5 trigram：查「巧克力」应命中「我喜欢酒心巧克力」。"""
        await self.store.add_long("我喜欢酒心巧克力，讨厌打针", importance=0.9, user="u1")
        await self.store.add_long("今天天气不错", importance=0.3, user="u1")

        hits = await self.store.search_long("巧克力", user="u1")
        self.assertTrue(any("酒心巧克力" in h for h in hits))
        self.assertFalse(any("天气" in h for h in hits))

    async def test_natural_question_recall(self):
        """自然语言长问句也能召回（查询被切成 3-gram 后 OR）。"""
        await self.store.add_long("我喜欢酒心巧克力，讨厌打针", importance=0.9, user="u1")
        hits = await self.store.search_long("我之前说过我喜欢吃什么", user="u1")
        self.assertTrue(any("酒心巧克力" in h for h in hits))

    async def test_short_query_falls_back_to_like(self):
        """少于 3 字（trigram 无法匹配）时走 LIKE 兜底。"""
        await self.store.add_long("早饭喜欢吃梨")
        self.assertTrue(await self.store.search_long("梨"))

    async def test_importance_recency_ranking(self):
        """高重要性 + 新的记忆排在前面。"""
        await self.store.add_long("低重要性的巧克力相关记忆", importance=0.1)
        await self.store.add_long("高重要性的巧克力相关记忆", importance=0.99)
        hits = await self.store.search_long("巧克力", top_k=1)
        self.assertEqual(hits, ["高重要性的巧克力相关记忆"])

    async def test_search_empty_db(self):
        self.assertEqual(await self.store.search_long("随便查点什么"), [])

    # ------------------------------------------------------------------ 画像
    async def test_profile_merge_and_dedupe(self):
        await self.store.upsert_profile("u", nick="小库", facts=["喜欢酒心巧克力"])
        await self.store.upsert_profile("u", facts=["喜欢酒心巧克力", "早饭吃梨"])
        profile = await self.store.get_profile("u")
        self.assertEqual(profile["nick"], "小库")            # 第二次没传 nick → 保留旧值
        self.assertEqual(profile["facts"], ["喜欢酒心巧克力", "早饭吃梨"])

    async def test_profile_default_for_unknown_user(self):
        profile = await self.store.get_profile("nobody")
        self.assertEqual(profile, {"nick": "", "facts": []})

    # ------------------------------------------------------------------ 会话 / 流水
    async def test_session_roundtrip_order(self):
        await self.store.add_session("较早的摘要", user="u", started_ts=1, ended_ts=2)
        await self.store.add_session("较新的摘要", user="u", started_ts=3, ended_ts=4)
        self.assertEqual(await self.store.recent_sessions(2, "u"),
                         ["较早的摘要", "较新的摘要"])

    async def test_message_roundtrip(self):
        await self.store.add_message("user", "你好", user="u", user_name="小明")
        await self.store.add_message("assistant", "你好呀", user="u")
        rows = await self.store.recent_messages(10, "u")
        self.assertEqual([r["role"] for r in rows], ["user", "assistant"])
        self.assertEqual(rows[0]["user_name"], "小明")

    async def test_stats(self):
        await self.store.add_message("user", "hi")
        await self.store.add_long("something", importance=0.5)
        await self.store.add_session("摘要", user="u", started_ts=0, ended_ts=1)
        stats = self.store.stats()
        self.assertEqual(stats["messages"], 1)
        self.assertEqual(stats["long_memory"], 1)
        self.assertEqual(stats["sessions"], 1)


if __name__ == "__main__":
    unittest.main()