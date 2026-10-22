"""插件生命周期测试（热重载 / 热插拔 / 删除 / Agent 自举工具 / 插件设置项）。

关键隔离手段：
    用临时目录替换 plugins_dir()，并把该目录挂进 ska.plugins.__path__，
    这样测试写出的插件不会污染仓库、也不会和真实插件互相干扰。
"""
from __future__ import annotations

import importlib
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import ska.plugins as plugins_pkg
from ska.kernel.bus import EventBus
from ska.kernel.event import TOPIC_WAKE
from ska.plugins import manager as pm
from ska.plugins import scheduler as scheduler_plugin
from ska.tools.base import ToolDeps


# 一个最小可用插件：装一个订阅 + 贡献一个工具
PLUGIN_SRC = '''
from ska.tools.base import Tool

class Demo(Tool):
    name = "demo_tool_x"
    async def run(self, **kw):
        return "ok"

def install(bus, cfg):
    bus.subscribe("output", lambda ev: None)      # 验证卸载时订阅会被摘掉
    class Plugin:
        def contribute_tools(self, deps):
            return {"demo_tool_x": Demo(deps)}
    return Plugin()
'''


class TestPluginManager(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        # 1) 把插件目录指到临时目录（manager 里所有路径都经 plugins_dir() 解析）
        self._patch = patch.object(pm, "plugins_dir", return_value=self.dir)
        self._patch.start()
        # 2) 让 importlib 也能在临时目录里找到模块
        self._orig_path = list(plugins_pkg.__path__)
        plugins_pkg.__path__.append(self._tmp.name)

        self.bus = EventBus()
        self.tools: dict = {}
        self.cfg = SimpleNamespace(get=lambda key, default=None: default)
        self.app: dict = {
            "bus": self.bus, "cfg": self.cfg, "tools": self.tools,
            "plugins": [], "plugin_map": {}, "tool_deps": ToolDeps(),
        }
        self.mgr = pm.PluginManager(self.app, settings=None, store=None)

    async def asyncTearDown(self) -> None:
        for name in list(self.app.get("plugin_map", {})):
            await self.mgr.uninstall(name)
            self.mgr._purge_modules(name)
        plugins_pkg.__path__[:] = self._orig_path
        self._patch.stop()
        self._tmp.cleanup()

    # ------------------------------------------------------------------ 基础
    async def test_create_installs_and_contributes_tool(self):
        result = await self.mgr.create("demo_x", PLUGIN_SRC)
        self.assertTrue(result["loaded"])
        self.assertTrue((self.dir / "demo_x.py").exists())
        self.assertIn("demo_tool_x", self.tools)            # 工具已合入
        self.assertIn("demo_x", self.mgr.loaded())

    async def test_uninstall_removes_tool_and_subscription(self):
        await self.mgr.create("demo_x", PLUGIN_SRC)
        self.assertEqual(len(self.bus._handlers.get("output", [])), 1)
        await self.mgr.uninstall("demo_x")
        self.assertNotIn("demo_tool_x", self.tools)         # 工具被回收
        self.assertEqual(len(self.bus._handlers.get("output", [])), 0)  # 订阅被摘掉
        self.assertNotIn("demo_x", self.mgr.loaded())

    async def test_reload_uses_new_source_and_no_duplicate_subscriptions(self):
        await self.mgr.create("demo_x", PLUGIN_SRC)
        new_src = PLUGIN_SRC.replace('return "ok"', 'return "v2"')
        self.mgr.write_source("demo_x", new_src)
        await self.mgr.reload("demo_x")

        self.assertEqual(len(self.bus._handlers.get("output", [])), 1)   # 不是 2
        tool = self.tools["demo_tool_x"]
        self.assertEqual(await tool.run(), "v2")            # 装载的确实是新代码

    async def test_write_rejects_bad_source(self):
        with self.assertRaises(pm.PluginError):
            self.mgr.write_source("bad_x", "def install(:\n  pass")     # 语法错误
        with self.assertRaises(pm.PluginError):
            self.mgr.write_source("bad_x", "x = 1")                     # 缺 install
        self.assertFalse((self.dir / "bad_x.py").exists())               # 失败不落盘

    async def test_name_validation_blocks_path_traversal(self):
        with self.assertRaises(pm.PluginError):
            self.mgr.write_source("../escape", PLUGIN_SRC)

    async def test_delete_removes_file(self):
        await self.mgr.create("demo_x", PLUGIN_SRC)
        await self.mgr.delete("demo_x")
        self.assertFalse((self.dir / "demo_x.py").exists())
        self.assertNotIn("demo_tool_x", self.tools)

    async def test_builtin_plugin_cannot_be_deleted(self):
        with self.assertRaises(pm.PluginError):
            await self.mgr.delete("filter")

    async def test_infrastructure_module_is_not_a_plugin(self):
        """manager.py 是插件基础设施，不能出现在清单里、也不能被覆写/删除。"""
        self.assertNotIn("manager", self.mgr.available())
        with self.assertRaises(pm.PluginError):
            self.mgr.write_source("manager", PLUGIN_SRC)
        with self.assertRaises(pm.PluginError):
            await self.mgr.delete("manager")

    async def test_install_is_idempotent(self):
        await self.mgr.create("demo_x", PLUGIN_SRC)
        again = await self.mgr.install("demo_x")
        self.assertIs(again, self.app["plugin_map"]["demo_x"])
        self.assertEqual(len(self.bus._handlers.get("output", [])), 1)

    # ------------------------------------------------------------------ 插件设置项
    async def test_scheduler_declares_targets_setting(self):
        """内置 scheduler 声明了「消息通道」schema，清单里应标记 has_settings。"""
        schema = self.mgr.settings_schema("scheduler")
        self.assertEqual(schema["title"], "定时任务")
        self.assertEqual(schema["fields"][0]["key"], "targets")
        self.assertEqual(schema["fields"][0]["type"], "targets")
        self.assertEqual(self.mgr.default_settings(schema), {"targets": []})
        entry = next(p for p in self.mgr.describe() if p["name"] == "scheduler")
        self.assertTrue(entry["has_settings"])

    async def test_validate_settings(self):
        """设置值按 schema 校验：未知键/坏目标报错，缺省键补默认值。"""
        good = ["qq|group|123", "qq|user|456"]
        self.assertEqual(self.mgr.validate_settings("scheduler", {"targets": good}),
                         {"targets": good})
        self.assertEqual(self.mgr.validate_settings("scheduler", {}), {"targets": []})
        with self.assertRaises(pm.PluginError):
            self.mgr.validate_settings("scheduler", {"nope": 1})
        with self.assertRaises(pm.PluginError):
            self.mgr.validate_settings("scheduler", {"targets": ["not-a-target"]})
        with self.assertRaises(pm.PluginError):
            self.mgr.validate_settings("demo_x", {"x": 1})     # 未声明设置项

    async def test_install_receives_settings_by_signature(self):
        """install 的第三个参数按签名注入：写 settings 参数的插件才拿得到。"""
        src = '''
RECEIVED = {}
def install(bus, cfg, settings=None):
    RECEIVED["settings"] = settings
    return None
'''
        marker = SimpleNamespace(names=lambda key: [], section=lambda name: {})
        self.mgr.settings = marker
        await self.mgr.create("demo_s", src)
        module = importlib.import_module("ska.plugins.demo_s")
        self.assertIs(module.RECEIVED["settings"], marker)

    # ------------------------------------------------------------------ Agent 自举工具
    async def test_plugin_ops_tool_can_self_bootstrap(self):
        from ska.tools.plugin_ops import PluginOpsTool
        tool = PluginOpsTool(ToolDeps(extra={"plugin_manager": self.mgr}))

        msg = await tool.run(action="write", name="demo_x", source=PLUGIN_SRC)
        self.assertIn("热装载", msg)
        self.assertIn("demo_tool_x", self.tools)

        listed = await tool.run(action="list")
        self.assertIn("demo_x", listed)

        await tool.run(action="disable", name="demo_x")
        self.assertNotIn("demo_x", self.mgr.loaded())

        await tool.run(action="enable", name="demo_x")
        self.assertIn("demo_x", self.mgr.loaded())

        await tool.run(action="delete", name="demo_x")
        self.assertFalse((self.dir / "demo_x.py").exists())

    async def test_plugin_ops_reports_missing_manager(self):
        from ska.tools.plugin_ops import PluginOpsTool
        tool = PluginOpsTool(ToolDeps())
        self.assertIn("插件管理器不可用", await tool.run(action="list"))


class _SchedCfg:
    """scheduler 插件所需的最小 Config 替身。"""

    def __init__(self, default_group: str = "") -> None:
        self.scheduler = {"tasks": []}
        self._qq = {"default_group": default_group}

    def get(self, key, default=None):
        return default

    def section(self, name: str) -> dict:
        return dict(self._qq) if name == "qq" else {}


class _SchedSettings:
    """最小 RuntimeSettings 替身：只返回插件自己的小节。"""

    def __init__(self, values: dict) -> None:
        self.values = values

    def section(self, name: str) -> dict:
        return dict(self.values) if name == "plugin:scheduler" else {}


class TestSchedulerTargets(unittest.IsolatedAsyncioTestCase):
    """scheduler 的消息通道设置：多选、默认群聊、触发时带上完整目标列表。"""

    def setUp(self) -> None:
        # 隔离：不读仓库里真实的任务表（否则测试会挂上真实日程）
        self._real_read_tasks = scheduler_plugin._read_tasks
        self._patch = patch.object(scheduler_plugin, "_read_tasks", return_value=[])
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()

    async def test_settings_targets_win_over_default_group(self):
        sched = scheduler_plugin.install(
            EventBus(), _SchedCfg("999"),
            _SchedSettings({"targets": ["qq|group|123", "qq|user|456"]}))
        self.assertEqual(sched.targets, [
            {"channel": "qq", "target_type": "group", "target_id": "123"},
            {"channel": "qq", "target_type": "user", "target_id": "456"},
        ])
        await sched.stop()

    async def test_defaults_to_qq_group(self):
        """没配消息通道时，默认发到 QQ 默认群（=群聊）。"""
        sched = scheduler_plugin.install(EventBus(), _SchedCfg("999"), _SchedSettings({}))
        self.assertEqual(sched.targets, [
            {"channel": "qq", "target_type": "group", "target_id": "999"}])
        await sched.stop()

    async def test_without_default_group_inherits_route(self):
        sched = scheduler_plugin.install(EventBus(), _SchedCfg(""), _SchedSettings({}))
        self.assertEqual(sched.targets, [])
        await sched.stop()

    async def test_read_tasks_filters_disabled_and_falls_back(self):
        """任务表从磁盘读取，只保留 enabled；读坏时退回启动快照。"""
        data = {"tasks": [{"name": "a", "at": "08:00", "enabled": True},
                          {"name": "b", "at": "09:00", "enabled": False},
                          "not-a-dict"]}
        with patch.object(scheduler_plugin, "load_yaml", return_value=data):
            self.assertEqual([t["name"] for t in self._real_read_tasks(_SchedCfg())], ["a"])
        with patch.object(scheduler_plugin, "load_yaml", side_effect=OSError("占用")):
            cfg = _SchedCfg()
            cfg.scheduler = {"tasks": [{"name": "c", "enabled": True}]}
            self.assertEqual([t["name"] for t in self._real_read_tasks(cfg)], ["c"])

    async def test_fire_carries_all_targets_and_first_route(self):
        bus = EventBus()
        got: list = []
        bus.subscribe(TOPIC_WAKE, got.append)
        targets = [{"channel": "qq", "target_type": "group", "target_id": "123"},
                   {"channel": "qq", "target_type": "user", "target_id": "456"}]
        sched = scheduler_plugin._Scheduler(bus, [], targets)
        with patch("ska.plugins.scheduler.asyncio.sleep", new=AsyncMock()):
            await sched._fire({"name": "morning", "payload": "起床"}, datetime.now())

        payload = got[0].payload
        self.assertEqual(payload.raw["_targets"], targets)      # 完整列表交给心跳扇出
        self.assertEqual((payload.channel, payload.message_type, payload.group_id),
                         ("qq", "group", "123"))                # 来路取第一个目标
        self.assertEqual(payload.task, "morning")
        self.assertEqual(payload.content, "起床")


if __name__ == "__main__":
    unittest.main()
