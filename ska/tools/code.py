"""编码智能体三级工具骨架（阶段三 · 方案 §5 / §3D）。

为什么单独成文件、且默认禁用：
    这三件能力（读文件 / 写文件 / 跑命令）是「让智能体通过代码与真实世界联动」的
    起点，但也是风险最高的一组工具——写错路径会毁数据、跑错命令会伤系统。所以本文件
    只提供**骨架 + 硬护栏**，默认全部关闭，需要管理员在 `config.yaml` 的 `code` 小节
    显式打开并**配置工作目录白名单**，再在「智能体权限」页把最小可用等级设为 admin。

三重护栏（缺一不可，任何一条不满足即拒绝执行）：
    ① 启用闸：`code.enabled` 默认 false → 直接拒绝（不依赖后台，配置即门槛）；
    ② 等级闸：工具类标注 min_level="admin"，由 Policy 的 tool_permissions 表真正生效
       （未在后台配置时，本文件的「启用闸 + 白名单闸」仍是兜底）；
    ③ 路径闸：所有路径必须落在 `code.allow_dirs` 白名单目录内（含子目录），越界即拒。

另加三条运行期约束：
    ④ 沙盒闸：`code.sandbox`（默认开）——清洗环境变量（只透传 PATH 等系统必需项，
       不把 *_API_KEY 之类私密变量交给命令），并用黑名单拦截典型破坏性命令；
    ⑤ 命令执行有超时（`code.timeout`）；
    ⑥ 所有输出统一截断（防爆窗/注入）。

注：本阶段是**骨架**，只做上面的保守实现；更丰富的「项目上下文注入」按方案放在
    `ska/plugins/` 插件位，不塞进内核。
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..kernel.config import package_root
from .base import Tool, decode_bytes, decode_output, tool

logger = logging.getLogger(__name__)

# 输出截断上限：工具结果会回填进对话上下文，过长既爆窗又易被 prompt 注入
MAX_OUTPUT = 4000
# 命令执行默认超时（秒）
DEFAULT_TIMEOUT = 10.0

# ===========================================================================
# 沙盒（sandbox）：在「白名单目录」之上再加两层阻力
# ---------------------------------------------------------------------------
# 目录白名单只保证「不越界读文件」，但 code_run 跑的是真·shell，风险更大，
# 所以再补两层（借鉴 open-interpreter / 各类 shell 沙盒的常见做法）：
#   ① 环境变量清洗：默认只透传最小可用环境（PATH 等系统必需项），
#      避免把 API key / 私密变量泄漏给被执行的命令；
#   ② 危险命令黑名单：命中典型的破坏性动作（rm -rf / mkfs / dd 写盘 / fork 炸弹…）直接拒绝。
# 注意：黑名单是「防御纵深」的最后一层、不是唯一防线（白名单目录才是主防线）；
#      正则应尽量「宁可误伤」——拒绝一条正常命令的代价，远低于放行一条破坏命令。
# ===========================================================================

# 危险命令黑名单：每项 (正则, 人类可读的理由)
_DANGEROUS_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brm\s+(-[a-z]+\s+)*-[a-z]*r[a-z]*f", "递归强制删除（rm -rf）"),
    (r"\bmkfs(\.[a-z0-9]+)?\b", "格式化文件系统（mkfs）"),
    (r"\bdd\s+if=", "裸写磁盘（dd if=）"),
    (r">\s*/dev/(sd|hd|nvme|disk)", "覆写块设备"),
    (r":\s*\(\s*\)\s*\{.*\}\s*;\s*:", "fork 炸弹"),
    (r"\b(shutdown|reboot|halt|poweroff)\b", "关机 / 重启系统"),
    (r"\b(format|diskpart)\b", "Windows 磁盘格式化"),
    (r"\bchmod\s+-[a-z]*R[a-z]*\s+777\s+/", "递归放开系统目录权限"),
    (r"\bdel\s+/[fsq]", "Windows 强制删除"),
    (r"Remove-Item\b.*-Recurse", "PowerShell 递归删除"),
)
_DANGEROUS_RE: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pat, re.I), why) for pat, why in _DANGEROUS_PATTERNS)

# 允许透传给被测命令的环境变量（其余一律清空）。
# * PATH 必需，否则找不到可执行文件；
# * Windows 下 SystemRoot / COMSPEC / PATHEXT / WINDIR 必需（很多内置命令依赖它们）；
# * LANG/TZ/PYTHONIOENCODING 用于保证输出文本可读、编码正确。
_BASE_ENV_KEYS: frozenset[str] = frozenset({
    "PATH", "HOME", "USERPROFILE", "LANG", "LC_ALL", "TZ",
    "SYSTEMROOT", "COMSPEC", "PATHEXT", "WINDIR",
    "TEMP", "TMP", "PYTHONIOENCODING", "PYTHONUTF8",
})


class _CodeToolBase(Tool):
    """三个编码工具的公共护栏：启用闸 / 路径白名单 / 沙盒 / 超时 / 截断。"""

    # 文档性标注：实际等级闸由 Policy.tool_permissions 表决定（见 policy.py）。
    # 这里保留成类属性，便于后台展示与本文件自解释。
    min_level = "admin"

    # ------------------------------------------------------------------ 配置
    def _cfg(self) -> dict[str, Any]:
        """读 `code` 小节（来自 config.yaml；缺 Config 时视为未配置 → 一律禁用）。"""
        cfg = getattr(self.deps, "cfg", None)
        section = getattr(cfg, "section", None)
        if not callable(section):
            return {}
        value = section("code")
        return value if isinstance(value, dict) else {}

    def _roots(self) -> list[Path]:
        """工作目录白名单（已 resolve 的绝对路径列表）。

        相对路径按**项目根**解析（而不是进程 cwd），这样配置写 `allow_dirs: ["."]`
        就稳定指向项目根，换个地方启动进程也不会走偏；绝对路径原样使用。
        """
        raw = self._cfg().get("allow_dirs") or []
        if isinstance(raw, str):
            raw = [raw]
        roots: list[Path] = []
        for item in raw:
            try:
                path = Path(str(item)).expanduser()
                if not path.is_absolute():
                    path = package_root() / path       # 相对路径 → 项目根
                roots.append(path.resolve())
            except (OSError, ValueError):  # 路径非法 → 跳过（不因此放行其它路径）
                logger.warning("忽略非法的 allow_dirs 项：%r", item)
        return roots

    def _timeout(self) -> float:
        try:
            return max(float(self._cfg().get("timeout", DEFAULT_TIMEOUT)), 1.0)
        except (TypeError, ValueError):
            return DEFAULT_TIMEOUT

    # ------------------------------------------------------------------ 沙盒
    def _sandbox_on(self) -> bool:
        """沙盒开关：默认开（`code.sandbox: false` 才关闭）。

        默认开是有意的——安全项应当「默认安全、显式关闭」，而不是反过来。
        """
        return bool(self._cfg().get("sandbox", True))

    def _safe_env(self) -> dict[str, str]:
        """构造一份「清洗过」的子进程环境变量。

        只保留白名单键（PATH 等系统必需项）+ `code.allow_env` 里显式追加的键，
        其余（含各类 *_API_KEY / *_TOKEN 等私密变量）一律不传给子进程。
        """
        allow = set(_BASE_ENV_KEYS)
        extra = self._cfg().get("allow_env") or []
        if isinstance(extra, str):
            extra = [extra]
        allow.update(str(k) for k in extra)
        env = {k: v for k, v in os.environ.items() if k in allow}
        # 两个变量一起设：PYTHONIOENCODING 管 stdout/stderr 编码，
        # PYTHONUTF8 让子进程的 open()/文件编码也走 UTF-8（Python 3.7+ 的 UTF-8 模式）。
        # 缺了后者，子进程「读中文文件」仍会按 GBK 打开，输出照样乱。
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        return env

    def _check_command(self, command: str) -> str | None:
        """危险命令检查：命中黑名单返回拒绝理由；None 表示放行。沙盒关闭时不做此检查。"""
        if not self._sandbox_on():
            return None
        for regex, why in _DANGEROUS_RE:
            if regex.search(command):
                logger.warning("沙盒拦截危险命令（%s）：%s", why, command[:120])
                return f"命令被沙盒拦截（疑似危险操作：{why}）：{command[:120]}"
        return None

    # ------------------------------------------------------------------ 护栏
    def _guard(self) -> str | None:
        """返回拒绝理由字符串；None 表示三道闸均通过、可以执行。"""
        cfg = self._cfg()
        if not bool(cfg.get("enabled", False)):
            return ("编码工具默认禁用：请在 config.yaml 的 code 小节设置 enabled: true，"
                    "并在 code.allow_dirs 配置允许访问的工作目录。")
        if not self._roots():
            return "未配置工作目录白名单：请在 config.yaml 的 code.allow_dirs 列出允许访问的目录。"
        return None

    def _resolve(self, path: str) -> Path:
        """把用户给的路径解析为绝对路径（相对路径相对于白名单第一项）。"""
        p = Path(str(path or "")).expanduser()
        if not p.is_absolute():
            roots = self._roots()
            p = (roots[0] if roots else Path.cwd()) / p
        return p.resolve()

    def _check_path(self, path: str) -> tuple[Path | None, str | None]:
        """把路径限制在白名单目录内；越界返回 (None, 错误文本)。"""
        target = self._resolve(path)
        for root in self._roots():
            if target == root or root in target.parents:
                return target, None
        return None, f"路径不在白名单内（已拒绝）：{target}"

    @staticmethod
    def _clip(text: str, limit: int = MAX_OUTPUT) -> str:
        """截断过长输出，并在末尾标注原始长度。"""
        text = text or ""
        return text if len(text) <= limit else text[:limit] + f"\n…（已截断，共 {len(text)} 字）"


# ===========================================================================
# code_read —— 读文件
# ===========================================================================
@tool("code_read")
class CodeReadTool(_CodeToolBase):
    description = ("读取工作目录白名单内某个文本文件的内容。"
                   "需要管理员级权限，且管理员已在 config.yaml 的 code 小节启用。")
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "文件路径（可为白名单内的相对路径）"},
            "max_lines": {"type": "integer", "description": "最多读取的行数，默认 400"},
        },
        "required": ["path"],
    }

    async def run(self, path: str = "", max_lines: int = 400, **_: Any) -> str:
        reason = self._guard()
        if reason:
            return reason
        target, err = self._check_path(path)
        if err:
            return err
        if not target.is_file():
            return f"不是文件或不存在：{target}"
        try:
            # 不假定文件一定是 UTF-8：Windows 记事本存的中文常是 GBK，
            # 直接 utf-8 读会得到一串替换符（表现为「文件里的中文全变问号/方块」）。
            text = decode_bytes(target.read_bytes())
        except OSError as e:
            return f"读取失败：{e}"
        lines = text.splitlines()
        limit = max(int(max_lines or 400), 1)
        if len(lines) > limit:
            lines = lines[:limit]
        return self._clip("\n".join(lines))


# ===========================================================================
# code_write —— 写文件（覆盖 / 追加）
# ===========================================================================
@tool("code_write")
class CodeWriteTool(_CodeToolBase):
    description = ("把内容写入工作目录白名单内的文件（默认覆盖，可选追加）。"
                   "需要管理员级权限，且管理员已在 config.yaml 的 code 小节启用。")
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "目标文件路径（白名单内）"},
            "content": {"type": "string", "description": "要写入的文本"},
            "append": {"type": "boolean", "description": "true=追加，false/缺省=覆盖"},
        },
        "required": ["path", "content"],
    }

    async def run(self, path: str = "", content: str = "", append: bool = False, **_: Any) -> str:
        reason = self._guard()
        if reason:
            return reason
        target, err = self._check_path(path)
        if err:
            return err
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a" if append else "w", encoding="utf-8") as fh:
                fh.write(content or "")
        except OSError as e:
            return f"写入失败：{e}"
        return f"已{'追加' if append else '写入'} {target}（{len(content or '')} 字）"


# ===========================================================================
# code_run —— 执行命令（带超时 + 输出截断）
# ===========================================================================
@tool("code_run")
class CodeRunTool(_CodeToolBase):
    description = ("在工作目录白名单内执行一条 shell 命令并返回输出（有超时与长度上限）。"
                   "需要管理员级权限，且管理员已在 config.yaml 的 code 小节启用。")
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的命令"},
            "cwd": {"type": "string", "description": "工作目录（白名单内，缺省用白名单第一项）"},
        },
        "required": ["command"],
    }

    async def run(self, command: str = "", cwd: str = "", **_: Any) -> str:
        reason = self._guard()
        if reason:
            return reason
        command = (command or "").strip()
        if not command:
            return "命令为空。"
        denied = self._check_command(command)      # 沙盒：危险命令黑名单
        if denied:
            return denied

        work = self._roots()[0]
        if cwd:
            work, err = self._check_path(cwd)   # type: ignore[assignment]
            if err:
                return err
        try:
            proc = await asyncio.create_subprocess_shell(
                command, cwd=str(work), env=self._safe_env(),  # 沙盒：清洗环境变量
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        except OSError as e:
            return f"命令启动失败：{e}"
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=self._timeout())
        except asyncio.TimeoutError:
            proc.kill()                          # 超时即杀，避免僵尸进程
            return f"命令超时（>{self._timeout():.0f}s）：{command}"
        # 逐行解码：cmd 自身的提示是 GBK，脚本输出多是 UTF-8，混在一起时
        # 整段解码必然有一半乱码；decode_output 按行各自挑编码，两边都能读。
        text = decode_output(out or b"")
        return f"退出码 {proc.returncode}\n{self._clip(text)}"
