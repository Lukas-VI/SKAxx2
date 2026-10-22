"""plugin_ops —— 让 Agent 给自己写插件的工具（自举用）。

为什么把它做成「工具」而不是只做后台按钮：
    本项目的 Agent 是「并行单实例」的自主体，希望它能在思考回环里自己扩展能力——
    发现缺一个能力 → 读现有插件作参考 → 写一个新插件 → 热装载 → 立刻用上。
    这条路径必须由 Agent 自己能走完，所以管理动作要暴露成 tool call。

安全边界（「一定范围内」自举）：
    * 只能操作 `ska/plugins/` 下的单文件插件（名字白名单校验，杜绝路径穿越）；
    * 写入前做语法 + install 入口校验，失败不落盘；
    * 内置插件（filter / scheduler）禁止删除，只允许重载/启停；
    * 工具本身标注 min_level=admin，由 Policy 的「智能体权限」表决定谁能调用。
"""
from __future__ import annotations

from typing import Any

from .base import Tool, tool

# 读源码时的返回上限：源码只用于「参考写法」，不必整份塞回上下文
MAX_SOURCE_READ = 8000


@tool("plugin_ops")
class PluginOpsTool(Tool):
    """插件自举工具：读写/重载/启停/删除插件。"""

    min_level = "admin"          # 与 code_* 同级的危险能力（会执行新代码）

    description = (
        "管理本机插件（可用来给自己扩展能力）。写插件前请先读插件目录下的 "
        "PLUGIN_GUIDE.md（含完整规范、事件主题、示例与常见坑）。"
        "action=list 列出插件及状态；"
        "action=read 读取某插件源码（学写法）；"
        "action=write 新建或覆写插件源码并立即热装载（必须提供 name 与 source，"
        "插件需定义 install(bus, cfg)）；"
        "action=reload 热重载（改完源码后使其生效）；"
        "action=enable/disable 热插拔；"
        "action=delete 卸载并删除插件文件（内置插件 filter/scheduler 不可删除）。"
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {"type": "string",
                       "enum": ["list", "read", "write", "reload",
                                "enable", "disable", "delete"]},
            "name": {"type": "string",
                     "description": "插件模块名（字母/数字/下划线，如 my_tool）"},
            "source": {"type": "string",
                       "description": "action=write 时的完整 Python 源码"},
            "enable": {"type": "boolean",
                       "description": "action=write 时是否立即启用（默认 true）"},
        },
        "required": ["action"],
    }

    async def run(self, action: str = "", name: str = "", source: str = "",
                  enable: bool = True, **_: Any) -> str:
        manager = (getattr(self.deps, "extra", None) or {}).get("plugin_manager")
        if manager is None:
            return "插件管理器不可用（当前装配未启用插件扩展）。"

        try:
            if action == "list":
                return self._format_list(manager.describe())
            if action == "read":
                text = manager.read_source(name)
                if len(text) > MAX_SOURCE_READ:
                    text = text[:MAX_SOURCE_READ] + f"\n…（已截断，原始 {len(text)} 字符）"
                return f"# {name}.py\n{text}"
            if action == "write":
                result = await manager.create(name, source, enabled=bool(enable))
                return (f"插件 {result['name']} 已写入并热装载"
                        f"（enabled={result['enabled']}, loaded={result['loaded']}）。")
            if action == "reload":
                await manager.reload(name)
                return f"插件 {name} 已热重载。"
            if action == "enable":
                await manager.toggle(name, True)
                return f"插件 {name} 已启用并装载。"
            if action == "disable":
                await manager.toggle(name, False)
                return f"插件 {name} 已停用并卸载。"
            if action == "delete":
                await manager.delete(name)
                return f"插件 {name} 已删除。"
            return f"未知操作：{action}"
        except Exception as e:  # noqa: BLE001 —— 工具层统一转成可读文本
            return f"插件操作失败：{e}"

    @staticmethod
    def _format_list(items: list[dict[str, Any]]) -> str:
        """把插件状态压缩成几行，便于模型快速判断「有没有、能不能改」。"""
        if not items:
            return "当前没有任何插件。"
        lines = []
        for item in items:
            flags = []
            if item.get("enabled"):
                flags.append("已启用")
            if item.get("loaded"):
                flags.append("运行中")
            if item.get("builtin"):
                flags.append("内置")
            lines.append(f"- {item['name']}：{'、'.join(flags) or '未启用'}")
        return "\n".join(lines)
