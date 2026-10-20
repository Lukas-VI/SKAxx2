"""MCP 接入测试（阶段三 §5 增强）。

用一个「假 MCP 服务器」脚本（标准库实现、stdio JSON-RPC）真跑一遍：
    * MCPClient：initialize 握手 → tools/list 发现 → tools/call 调用；
    * extract_text：把 content 分片拼成可读文本，并识别 isError；
    * MCPTool / MCPManager：把远端工具包装成本机 Tool 并合入工具表。
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

from ska.mcp.bridge import MCPManager, MCPTool
from ska.mcp.client import MCPClient, extract_text

# --------------------------------------------------------------------------- 假服务器
# 一个最小的 MCP 服务器：只暴露一个 echo 工具，完整走 initialize / tools/list / tools/call。
FAKE_SERVER = '''\
import json
import sys

TOOLS = [{
    "name": "echo",
    "description": "回声工具",
    "inputSchema": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}]


def send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\\n")
    sys.stdout.flush()


while True:
    line = sys.stdin.readline()          # 逐行读，避免 for-in-stdin 的读缓冲问题
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except Exception:
        continue
    method = msg.get("method")
    mid = msg.get("id")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake", "version": "1.0"}}})
    elif method == "notifications/initialized":
        pass
    elif method == "tools/list":
        send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
    elif method == "tools/call":
        args = (msg.get("params") or {}).get("arguments") or {}
        send({"jsonrpc": "2.0", "id": mid, "result": {
            "content": [{"type": "text", "text": "echo: " + str(args.get("text", ""))}],
            "isError": False}})
    elif mid is not None:
        send({"jsonrpc": "2.0", "id": mid, "result": {}})
'''


class _StubCfg:
    """Config 最小替身：只需 section("mcp")。"""

    def __init__(self, data: dict | None = None) -> None:
        self._d = data or {}

    def section(self, name: str) -> dict:
        value = self._d.get(name, {})
        return value if isinstance(value, dict) else {}


class _FakeServerMixin:
    """把假服务器脚本落到临时目录，返回可用于启动的命令行。"""

    def _make_server(self) -> str:
        self._tmp = tempfile.TemporaryDirectory()
        script = os.path.join(self._tmp.name, "fake_mcp_server.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(FAKE_SERVER)
        return script


# --------------------------------------------------------------------------- 工具函数
class TestExtractText(unittest.TestCase):
    def test_join_text_parts(self):
        self.assertEqual(extract_text({"content": [{"type": "text", "text": "a"},
                                                   {"type": "text", "text": "b"}]}),
                         "a\nb")

    def test_error_flagged(self):
        out = extract_text({"isError": True, "content": [{"type": "text", "text": "坏了"}]})
        self.assertTrue(out.startswith("[MCP 返回错误]"))

    def test_non_dict_fallback(self):
        self.assertEqual(extract_text("你好"), "你好")


# --------------------------------------------------------------------------- 客户端
class TestMCPClient(_FakeServerMixin, unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        tmp = getattr(self, "_tmp", None)
        if tmp is not None:
            tmp.cleanup()

    async def test_handshake_list_and_call(self):
        script = self._make_server()
        client = MCPClient("fake", sys.executable, args=[script], timeout=10)
        await client.start()
        try:
            tools = await client.list_tools()
            self.assertEqual(len(tools), 1)
            self.assertEqual(tools[0]["name"], "echo")
            out = await client.call_tool("echo", {"text": "你好"})
            self.assertIn("echo: 你好", out)
        finally:
            await client.stop()

    async def test_start_failure_raises(self):
        """命令不存在 → start() 抛 MCPError，而不是留个半死状态。"""
        from ska.mcp.client import MCPError
        client = MCPClient("nope", "definitely-not-a-real-command-xyz", timeout=5)
        with self.assertRaises(MCPError):
            await client.start()


# --------------------------------------------------------------------------- 桥接/管理器
class TestMCPManager(_FakeServerMixin, unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self) -> None:
        tmp = getattr(self, "_tmp", None)
        if tmp is not None:
            tmp.cleanup()

    def test_from_config_disabled(self):
        self.assertEqual(MCPManager.from_config(_StubCfg({"mcp": {"enabled": False}})).tools, {})
        self.assertEqual(MCPManager.from_config(_StubCfg({})).tools, {})

    def test_tool_wrapping_name_and_schema(self):
        """MCPTool 的命名带前缀，参数直接取自 inputSchema。"""
        class _DummyClient:
            name = "srv"

        spec = {"name": "do", "description": "干活的",
                "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}}}
        tool = MCPTool(_DummyClient(), "do", spec)
        self.assertEqual(tool.name, "mcp__srv__do")
        self.assertEqual(tool.description, "干活的")
        self.assertEqual(tool.min_level, "admin")
        self.assertEqual(tool.parameters["properties"]["x"]["type"], "string")

    async def test_connect_and_inject(self):
        """连上假服务器 → 发现 echo → 合入工具表 → 能真的调用。"""
        script = self._make_server()
        cfg = _StubCfg({"mcp": {"enabled": True, "timeout": 10, "servers": [
            {"name": "fake", "command": sys.executable, "args": [script]}]}})
        manager = MCPManager.from_config(cfg)
        await manager.connect_all()
        try:
            self.assertIn("mcp__fake__echo", manager.tools)
            target: dict = {}
            self.assertEqual(manager.inject(target), 1)
            self.assertIn("mcp__fake__echo", target)
            out = await target["mcp__fake__echo"].run(text="hi")
            self.assertIn("echo: hi", out)
        finally:
            await manager.stop()

    async def test_bad_server_is_skipped(self):
        """单个服务器启动失败只跳过，不影响管理器返回。"""
        cfg = _StubCfg({"mcp": {"enabled": True, "timeout": 3, "servers": [
            {"name": "bad", "command": "definitely-not-a-real-command-xyz"}]}})
        manager = MCPManager.from_config(cfg)
        await manager.connect_all()          # 不应抛异常
        self.assertEqual(manager.tools, {})


# --------------------------------------------------------------------------- 插件贡献工具
class TestPluginContributedTools(unittest.TestCase):
    """插件以 contribute_tools(deps) 扩展工具表（与 MCP 同一张表，阶段三增强）。"""

    def test_plugin_tools_merged(self):
        from main import _collect_plugin_tools

        class _Tool:
            name = "plug_tool"
            description = "插件贡献的工具"

            async def run(self, **kwargs):
                return "ok"

            def schema(self):
                return {"type": "function",
                        "function": {"name": self.name, "description": "",
                                     "parameters": {}}}

        class _Plugin:
            def contribute_tools(self, deps):
                self.seen_deps = deps
                return {"plug_tool": _Tool()}

        app = {"tools": {"time": object()}, "tool_deps": {"x": 1}}
        plugin = _Plugin()
        _collect_plugin_tools(app, plugin)
        self.assertIn("plug_tool", app["tools"])        # 新工具已合入工具表
        self.assertEqual(plugin.seen_deps, {"x": 1})    # deps 透传给了插件

    def test_conflict_keeps_existing(self):
        from main import _collect_plugin_tools

        class _Plugin:
            def contribute_tools(self, deps):
                return {"time": object()}               # 与内置工具同名

        original = object()
        app = {"tools": {"time": original}, "tool_deps": None}
        _collect_plugin_tools(app, _Plugin())
        self.assertIs(app["tools"]["time"], original)   # 冲突时以既有工具为准

    def test_plugin_without_hook_is_noop(self):
        from main import _collect_plugin_tools

        class _Plugin:
            pass

        app = {"tools": {}}
        _collect_plugin_tools(app, _Plugin())           # 无钩子 → 不抛异常
        self.assertEqual(app["tools"], {})


if __name__ == "__main__":
    unittest.main()
