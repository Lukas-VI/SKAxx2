"""编码智能体三级工具测试（阶段三 §5 / 3D）。

重点验证「默认禁用 + 三重护栏」是否真的挡住越权操作：
    * 未启用 / 未配白名单 → 一律拒绝；
    * 白名单内的读写 → 正常且不越界；
    * 路径逃逸 → 被拒；
    * 命令执行 → 有输出与退出码；
    * 沙盒（阶段三增强）→ 清洗环境变量 + 危险命令黑名单。
"""
from __future__ import annotations

import os
import tempfile
import unittest

from ska.tools.base import ToolDeps
from ska.tools.code import CodeReadTool, CodeRunTool, CodeWriteTool


class _StubCfg:
    """Config 最小替身：只需 section("code")。"""

    def __init__(self, data: dict | None = None) -> None:
        self._d = data or {}

    def section(self, name: str) -> dict:
        value = self._d.get(name, {})
        return value if isinstance(value, dict) else {}


class TestCodeTools(unittest.IsolatedAsyncioTestCase):
    def _tool(self, cls, code: dict):
        return cls(ToolDeps(cfg=_StubCfg({"code": code})))

    def test_min_level_marked_admin(self):
        self.assertEqual(CodeReadTool.min_level, "admin")
        self.assertEqual(CodeWriteTool.min_level, "admin")
        self.assertEqual(CodeRunTool.min_level, "admin")

    async def test_disabled_by_default(self):
        t = self._tool(CodeReadTool, {})
        self.assertIn("默认禁用", await t.run(path="x.txt"))

    async def test_requires_allow_dirs(self):
        t = self._tool(CodeReadTool, {"enabled": True, "allow_dirs": []})
        self.assertIn("白名单", await t.run(path="x.txt"))

    async def test_read_write_within_whitelist(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = {"enabled": True, "allow_dirs": [tmp]}
            write = self._tool(CodeWriteTool, code)
            self.assertIn("已写入", await write.run(path="a/b.txt", content="你好"))
            read = self._tool(CodeReadTool, code)
            self.assertEqual(await read.run(path="a/b.txt"), "你好")

    async def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = {"enabled": True, "allow_dirs": [tmp]}
            read = self._tool(CodeReadTool, code)
            out = await read.run(path="../../etc/passwd")
            self.assertIn("不在白名单", out)

    async def test_run_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = {"enabled": True, "allow_dirs": [tmp], "timeout": 5}
            t = self._tool(CodeRunTool, code)
            out = await t.run(command="echo hello")
            self.assertIn("hello", out)
            self.assertIn("退出码 0", out)

    # ---------------------------------------------------------------- 沙盒
    async def test_sandbox_blocks_dangerous_command(self):
        """危险命令（rm -rf）在沙盒开启时被直接拦截，不会真的执行。"""
        with tempfile.TemporaryDirectory() as tmp:
            t = self._tool(CodeRunTool, {"enabled": True, "allow_dirs": [tmp]})
            out = await t.run(command="rm -rf /")
            self.assertIn("沙盒", out)

    async def test_sandbox_sanitizes_env(self):
        """私密环境变量不外泄给被测命令；系统必需的 PATH 予以保留。"""
        os.environ["SKA_TEST_SECRET"] = "topsecret"
        try:
            t = self._tool(CodeRunTool, {"enabled": True, "allow_dirs": [os.getcwd()]})
            env = t._safe_env()
            self.assertNotIn("SKA_TEST_SECRET", env)
            self.assertIn("PATH", env)
        finally:
            os.environ.pop("SKA_TEST_SECRET", None)

    async def test_sandbox_allow_env_extends_whitelist(self):
        """白名单外的变量默认剔除；经 allow_env 显式声明的可透传。"""
        os.environ["SKA_TEST_KEEP"] = "ok"
        try:
            t = self._tool(CodeRunTool, {"enabled": True, "allow_dirs": [os.getcwd()],
                                         "allow_env": ["SKA_TEST_KEEP"]})
            self.assertEqual(t._safe_env().get("SKA_TEST_KEEP"), "ok")
        finally:
            os.environ.pop("SKA_TEST_KEEP", None)

    async def test_sandbox_off_disables_blacklist(self):
        """关闭沙盒后黑名单不生效（此处只验证「不拦截」，不真的执行破坏命令）。"""
        t = self._tool(CodeRunTool, {"enabled": True, "allow_dirs": [os.getcwd()],
                                     "sandbox": False})
        self.assertIsNone(t._check_command("rm -rf /"))


if __name__ == "__main__":
    unittest.main()
