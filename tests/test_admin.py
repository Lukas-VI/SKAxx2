"""后台层测试：管理库 CRUD、鉴权、权限策略、运行时设置覆盖层。

阶段二新增模块的单测（方案 §6 第 5 批要求）。全部用临时目录里的 sqlite，
不触碰 data/admin.db。
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest

from ska.admin.policy import LEVELS, Policy, level_rank
from ska.admin.settings import RuntimeSettings
from ska.admin.store import AdminStore
from ska.kernel.packet import CoreInput
from ska.web.auth import AuthService, hash_password, role_at_least, verify_password


# --------------------------------------------------------------------------- 桩
class _StubCfg:
    """Config 的最小替身：只提供 RuntimeSettings 用到的三个接口。"""

    def __init__(self, data: dict | None = None) -> None:
        self.cfg = data or {}

    def get(self, key, default=None):
        return self.cfg.get(key, default)

    def section(self, name):
        value = self.cfg.get(name, {})
        return value if isinstance(value, dict) else {}


class _StubSettings:
    """只实现 Policy 需要的方法（roles/agent），避免依赖真实 config.yaml。"""

    def __init__(self, roles: dict | None = None, agent: dict | None = None) -> None:
        self._roles = roles or {"owner_user": "", "default_level": "member"}
        self._agent = agent or {"allow_tools": True, "max_steps": 5}

    def roles(self):
        return dict(self._roles)

    def agent(self):
        return dict(self._agent)


class _FakeTool:
    """占位工具对象：Policy 只关心工具名，不调用其方法。"""

    def __init__(self, name: str) -> None:
        self.name = name


# =========================================================================== 管理库
class TestAdminStore(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = AdminStore(os.path.join(self._tmp.name, "admin.db"))

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self._tmp.cleanup()

    # ------------------------------------------------------------------ 账号 / 会话
    async def test_admin_crud(self):
        self.assertEqual(await self.store.count_admins(), 0)
        admin_id = await self.store.create_admin("admin", "h", "s", "owner")
        self.assertEqual(await self.store.count_admins(), 1)

        row = await self.store.get_admin("admin")
        self.assertEqual(row["role"], "owner")
        self.assertNotIn("pass_hash", (await self.store.list_admins())[0])

        self.assertTrue(await self.store.update_admin(admin_id, role="admin"))
        self.assertEqual((await self.store.get_admin("admin"))["role"], "admin")
        self.assertTrue(await self.store.delete_admin(admin_id))
        self.assertIsNone(await self.store.get_admin("admin"))

    async def test_session_expiry(self):
        await self.store.create_session("tok", "admin", time.time() + 60)
        self.assertIsNotNone(await self.store.get_session("tok"))
        await self.store.create_session("old", "admin", time.time() - 1)
        self.assertIsNone(await self.store.get_session("old"))    # 过期即清理
        await self.store.delete_session("tok")
        self.assertIsNone(await self.store.get_session("tok"))

    # ------------------------------------------------------------------ 设置
    async def test_settings_roundtrip(self):
        await self.store.save_settings({"app": {"bpm": 12}, "qq": {"enabled": True}})
        loaded = self.store.all_settings_sync()
        self.assertEqual(loaded["app"]["bpm"], 12)
        self.assertTrue(loaded["qq"]["enabled"])

    # ------------------------------------------------------------------ 用量
    async def test_usage_summary(self):
        await self.store.record_usage("p1", "m1", "chat", 10, 5, 15)
        await self.store.record_usage("p1", "m1", "chat", 20, 10, 30)
        summary = await self.store.usage_summary(days=7, group_by="provider")
        self.assertEqual(summary["totals"]["calls"], 2)
        self.assertEqual(summary["totals"]["total_tokens"], 45)
        self.assertEqual(summary["rows"][0]["group"], "p1")

    # ------------------------------------------------------------------ 聊天用户 / 目标 / 工具
    async def test_chat_user_touch_and_update(self):
        await self.store.touch_chat_user("qq", "10001", "小明")
        await self.store.touch_chat_user("qq", "10001", "小明")
        row = await self.store.get_chat_user("qq", "10001")
        self.assertEqual(row["msg_count"], 2)
        await self.store.update_chat_user("qq", "10001", level="guest", blacklisted=True)
        row = await self.store.get_chat_user("qq", "10001")
        self.assertEqual(row["level"], "guest")
        self.assertEqual(row["blacklisted"], 1)

    async def test_publish_targets(self):
        await self.store.add_target("qq", "group", "999", "测试群")
        await self.store.add_target("qq", "group", "999", "测试群改")   # 幂等 upsert
        rows = await self.store.list_targets()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "测试群改")
        self.assertTrue(await self.store.update_target(rows[0]["id"], enabled=False))
        self.assertFalse((await self.store.list_targets())[0]["enabled"])
        self.assertTrue(await self.store.delete_target(rows[0]["id"]))

    async def test_tool_permissions(self):
        await self.store.upsert_tool_permission("fetch", True, "admin", "")     # 首次插入
        await self.store.upsert_tool_permission("fetch", False, "owner", "禁用")  # 冲突更新
        perms = {p["tool"]: p for p in await self.store.list_tool_permissions()}
        self.assertEqual(perms["fetch"]["min_level"], "owner")
        self.assertEqual(perms["fetch"]["enabled"], 0)

    async def test_stats(self):
        await self.store.create_admin("a", "h", "s", "owner")
        stats = await self.store.stats()
        self.assertEqual(stats["admin_users"], 1)
        self.assertEqual(stats["usage"], 0)


# =========================================================================== 鉴权
class TestAuth(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = AdminStore(os.path.join(self._tmp.name, "admin.db"))
        self.auth = AuthService(self.store, session_hours=72)

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self._tmp.cleanup()

    def test_password_hash_verify(self):
        h, salt = hash_password("hunter2")
        self.assertTrue(verify_password("hunter2", h, salt))
        self.assertFalse(verify_password("wrong", h, salt))
        self.assertFalse(verify_password("hunter2", "", salt))     # 空哈希不安全放行

    def test_role_rank(self):
        self.assertTrue(role_at_least("owner", "admin"))
        self.assertFalse(role_at_least("viewer", "admin"))
        self.assertTrue(role_at_least("admin", "admin"))

    async def test_setup_then_login(self):
        self.assertTrue(await self.auth.needs_setup())
        result = await self.auth.setup("admin", "secret123")
        self.assertEqual(result["role"], "owner")
        self.assertFalse(await self.auth.needs_setup())
        # setup 已直接签发会话
        self.assertIsNotNone(await self.auth.resolve(result["token"]))
        # 重复初始化应被拒绝
        with self.assertRaises(RuntimeError):
            await self.auth.setup("admin2", "secret123")

    async def test_login_flow(self):
        await self.auth.setup("admin", "secret123")
        self.assertIsNone(await self.auth.login("admin", "bad"))         # 口令错
        good = await self.auth.login("admin", "secret123")
        self.assertIsNotNone(good)
        resolved = await self.auth.resolve(good["token"])
        self.assertEqual(resolved["username"], "admin")
        await self.auth.logout(good["token"])
        self.assertIsNone(await self.auth.resolve(good["token"]))         # 登出后失效

    async def test_resolve_invalid_token(self):
        self.assertIsNone(await self.auth.resolve(""))
        self.assertIsNone(await self.auth.resolve("not-a-token"))


# =========================================================================== 运行时设置
class TestRuntimeSettings(unittest.TestCase):
    def _settings(self, data: dict | None = None) -> RuntimeSettings:
        return RuntimeSettings(_StubCfg(data or {}))

    def test_defaults_and_overrides(self):
        s = self._settings({"bpm": 6, "qq": {"api_base": "http://x:3000"}})
        s.load_sync({})
        self.assertEqual(s.get("bpm"), 6)
        self.assertEqual(s.section("qq")["api_base"], "http://x:3000")    # config.yaml 初值
        self.assertFalse(s.section("qq")["enabled"])                     # 内置默认兜底

        s.load_sync({"app": {"bpm": 12}, "qq": {"enabled": True}})
        self.assertEqual(s.get("bpm"), 12)                                # 覆盖层优先
        self.assertTrue(s.section("qq")["enabled"])
        self.assertEqual(s.snapshot()["app"]["bpm"], 12)

    def test_names_and_persona_override(self):
        s = self._settings({"tools": ["time"], "persona": {"profile": "local_full"}})
        s.load_sync({"app": {"persona_profile": "cloud_safe"}})
        self.assertEqual(s.persona_profile(), "cloud_safe")
        # 覆盖档位应写回 Config，让 build_system_prompt 也生效
        self.assertEqual(s.cfg.cfg["persona"]["profile"], "cloud_safe")
        self.assertEqual(s.names("tools"), ["time"])
        self.assertEqual(s.names("plugins"), [])

    def test_unknown_sections_filtered(self):
        s = self._settings({})
        s.load_sync({"evil": {"x": 1}, "app": {"bpm": 3}})
        self.assertNotIn("evil", s.overrides())
        self.assertEqual(s.overrides()["app"]["bpm"], 3)


# =========================================================================== 权限策略
class TestPolicy(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = AdminStore(os.path.join(self._tmp.name, "admin.db"))
        self.tools = {"time": _FakeTool("time"), "fetch": _FakeTool("fetch")}

    async def asyncTearDown(self) -> None:
        await self.store.close()
        self._tmp.cleanup()

    def _policy(self, **kw) -> Policy:
        return Policy(self.store, _StubSettings(**kw), self.tools)

    @staticmethod
    def _msg(user: str = "u1", channel: str = "qq") -> CoreInput:
        return CoreInput(source="qq_message", content="hi", user=user, channel=channel)

    def test_level_rank(self):
        self.assertGreater(level_rank("owner"), level_rank("member"))
        self.assertEqual(level_rank("已删等级"), 0)     # 未知等级按最低处理

    async def test_level_of_default_and_owner(self):
        p = self._policy()
        self.assertEqual(await p.level_of("qq", "u1"), "member")           # 默认
        p2 = self._policy(roles={"owner_user": "999", "default_level": "guest"})
        self.assertEqual(await p2.level_of("qq", "999"), "owner")          # 钦定 owner
        self.assertEqual(await p2.level_of("qq", "u1"), "guest")           # 覆盖默认

        await self.store.touch_chat_user("qq", "u2", "某人")
        await self.store.update_chat_user("qq", "u2", level="admin")
        self.assertEqual(await p.level_of("qq", "u2"), "admin")            # 表内记录

    async def test_blacklist_gate(self):
        p = self._policy()
        await self.store.touch_chat_user("qq", "u1", "坏蛋")
        await self.store.update_chat_user("qq", "u1", blacklisted=True)
        self.assertTrue(await p.is_blacklisted("qq", "u1"))
        self.assertFalse(await p.should_respond(self._msg("u1")))
        self.assertTrue(await p.should_respond(CoreInput(source="heartbeat")))  # 非用户消息放行

    async def test_note_user_records(self):
        p = self._policy()
        await p.note_user(self._msg("u7"))
        self.assertIsNotNone(await self.store.get_chat_user("qq", "u7"))

    async def test_allowed_tools_default_then_configured(self):
        p = self._policy()
        self.assertEqual(await p.allowed_tools("member"), {"time", "fetch"})   # 未配置默认放行

        await self.store.upsert_tool_permission("fetch", True, "admin")
        self.assertEqual(await p.allowed_tools("member"), {"time"})            # 等级不够
        self.assertEqual(await p.allowed_tools("admin"), {"time", "fetch"})    # 等级够
        self.assertEqual(await p.allowed_tools("owner"), {"time", "fetch"})

        await self.store.upsert_tool_permission("fetch", False, "member")      # 显式关闭
        self.assertEqual(await p.allowed_tools("owner"), {"time"})

    async def test_allowed_tools_global_switch(self):
        p = self._policy(agent={"allow_tools": False, "max_steps": 5})
        self.assertEqual(await p.allowed_tools("owner"), set())

    async def test_tools_for_scopes_by_user(self):
        p = self._policy()
        self.assertIsNone(await p.tools_for(CoreInput(source="heartbeat")))   # 非用户消息不裁剪
        self.assertEqual(await p.tools_for(self._msg()), {"time", "fetch"})

    async def test_targets_and_can_publish(self):
        p = self._policy()
        # 未配置白名单 → 一切放行（D7：保持阶段一行为）
        self.assertTrue(await p.can_publish("qq", "group", "999"))
        self.assertEqual(await p.targets("qq"), [])

        await self.store.add_target("qq", "group", "999", "群A", enabled=True)
        await self.store.add_target("qq", "user", "888", "好友B", enabled=False)
        self.assertTrue(await p.can_publish("qq", "group", "999"))
        self.assertFalse(await p.can_publish("qq", "user", "888"))      # 已禁用
        self.assertFalse(await p.can_publish("qq", "group", "111"))     # 不在白名单
        names = [t["target_id"] for t in await p.targets("qq")]
        self.assertEqual(names, ["999"])                                # 只列启用的


if __name__ == "__main__":
    unittest.main()