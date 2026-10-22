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

    # ------------------------------------------------------------------ 结构化事实（Mem0 风格）
    async def test_fact_upsert_dedupes_same_slot(self):
        """同一 (subject, predicate) 是同一槽位：新值覆盖旧值，不产生重复行。"""
        self.assertEqual(await self.store.upsert_fact("对方", "喜欢", "酒心巧克力"), "added")
        self.assertEqual(await self.store.upsert_fact("对方", "喜欢", "牛奶巧克力"), "updated")
        facts = await self.store.list_facts()
        self.assertEqual(len(facts), 1)                         # 去重：只剩一行
        self.assertEqual(facts[0]["object"], "牛奶巧克力")       # 冲突消解：新值覆盖旧值
        self.assertEqual(facts[0]["updated_count"], 2)          # 更新次数累加

    async def test_fact_importance_takes_max(self):
        """同槽位合并时重要度取较高者（不让低置信的一次覆盖高置信的）。"""
        await self.store.upsert_fact("对方", "喜欢", "a", importance=0.9)
        await self.store.upsert_fact("对方", "喜欢", "b", importance=0.2)
        facts = await self.store.list_facts()
        self.assertEqual(facts[0]["importance"], 0.9)

    async def test_fact_search_by_relevance(self):
        """结构化事实检索按查询 n-gram 相关度召回（中文友好，无需分词）。"""
        await self.store.upsert_fact("对方", "喜欢", "酒心巧克力", importance=0.6)
        await self.store.upsert_fact("对方", "住在", "城东", importance=0.6)   # 不同槽位
        self.assertEqual(await self.store.search_facts("巧克力", top_k=1),
                         ["对方喜欢酒心巧克力"])

    # ------------------------------------------------------------------ 核心记忆（MemGPT core 风格）
    async def test_core_pin_list_unpin(self):
        """核心记忆：pin 幂等、list 可读、unpin 可删。"""
        await self.store.pin_core("对方叫我小赛")
        await self.store.pin_core("对方叫我小赛")               # 重复 pin → 幂等
        self.assertEqual(await self.store.list_core(), ["对方叫我小赛"])
        self.assertTrue(await self.store.unpin_core("对方叫我小赛"))
        self.assertEqual(await self.store.list_core(), [])

    async def test_core_global_visible_to_any_user(self):
        """user='' 写入的核心记忆，对带 user 的检索也可读（单用户陪伴场景）。"""
        await self.store.pin_core("全局核心记忆", user="")
        self.assertEqual(await self.store.list_core(user="u1"), ["全局核心记忆"])

    async def test_list_core_entries_and_delete_by_id(self):
        """后台列表要拿到整行（id/重要性/归属/时间），并支持按 id 删除。"""
        core_id = await self.store.pin_core("对方叫我小赛", user="u1", importance=0.9)
        rows = await self.store.list_core_entries(user="u1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], core_id)
        self.assertEqual(rows[0]["importance"], 0.9)
        self.assertEqual(rows[0]["user"], "u1")
        self.assertTrue(await self.store.delete_core(core_id))
        self.assertEqual(await self.store.list_core_entries(user="u1"), [])

    async def test_delete_fact_by_id(self):
        """结构化事实支持按 id 删除（后台纠错）。"""
        await self.store.upsert_fact("对方", "喜欢", "酒心巧克力")
        facts = await self.store.list_facts()
        self.assertTrue(await self.store.delete_fact(facts[0]["id"]))
        self.assertEqual(await self.store.list_facts(), [])

    # ------------------------------------------------------------------ 记忆块组装（三层）
    async def test_recall_block_includes_core_and_facts(self):
        """记忆块应同时包含「始终记得的核心记忆」与「检索到的结构化事实」。"""
        from ska.memory.recall import build_memory_block

        await self.store.pin_core("对方叫我小赛")
        await self.store.upsert_fact("对方", "喜欢", "酒心巧克力", importance=0.7)
        block = await build_memory_block(self.store, "我最近喜欢吃什么", user="")
        self.assertIn("你一直记着的事", block)                  # 核心记忆常驻
        self.assertIn("小赛", block)
        self.assertIn("酒心巧克力", block)                      # 结构化事实被召回

    async def test_recall_block_empty_when_nothing(self):
        from ska.memory.recall import build_memory_block

        self.assertEqual(await build_memory_block(self.store, "", user=""), "")

    # ------------------------------------------------------------------ memory_ops 工具
    async def test_memory_ops_pin_and_unpin(self):
        """memory_ops 工具支持 pin/unpin，把内容提升/取消为核心记忆。"""
        from ska.tools.base import ToolDeps
        from ska.tools.builtin import MemoryOpsTool

        tool = MemoryOpsTool(ToolDeps(store=self.store))
        msg = await tool.run(action="pin", content="对方讨厌打针")
        self.assertIn("核心", msg)
        self.assertEqual(await self.store.list_core(), ["对方讨厌打针"])
        await tool.run(action="unpin", content="对方讨厌打针")
        self.assertEqual(await self.store.list_core(), [])


if __name__ == "__main__":
    unittest.main()