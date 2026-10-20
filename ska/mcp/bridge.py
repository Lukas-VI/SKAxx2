"""MCP 桥接层：把外部 MCP 服务器发现的工具，包装成内核认识的本机 Tool。

为什么需要「桥」：
    内核只认 Tool 接口（name / description / parameters / run）。MCP 工具的形状
    与之不同（远端名字 + inputSchema + 需要走 JSON-RPC 调用），所以用一个瘦适配器
    MCPTool 抹平差异，让 AgentLoop / Policy 完全无需知道「这工具其实在别的进程里」。

命名与权限（两条保守约定）：
    * 前缀 `mcp__<服务器>__<工具>`：避免与内置工具（time / code_run...）重名，
      也让模型与日志一眼看出工具来自哪个外部服务器；
    * 默认按 admin 级对待：外部工具的能力不可预知，按最高风险处理，需在
      后台「智能体权限」页显式放行 —— 安全项默认收紧。
"""
from __future__ import annotations

import logging
from typing import Any

from ..tools.base import Tool, ToolDeps
from .client import MCPClient, MCPError

logger = logging.getLogger(__name__)

NAME_PREFIX = "mcp__"
# MCP 工具默认要求的最小可用等级（见 admin/policy.py 的三级权限模型）
DEFAULT_MIN_LEVEL = "admin"


class MCPTool(Tool):
    """一个远端 MCP 工具在本机的「代理实现」。"""

    min_level = DEFAULT_MIN_LEVEL

    def __init__(self, client: MCPClient, remote_name: str, spec: dict[str, Any], *,
                 name_prefix: str = NAME_PREFIX, deps: ToolDeps | None = None) -> None:
        super().__init__(deps)
        self._client = client                       # 归属的服务器连接
        self._remote = remote_name                  # 服务器上的原始工具名
        self.name = f"{name_prefix}{client.name}__{remote_name}"
        self.description = str(spec.get("description")
                               or f"由 MCP 服务器 {client.name} 提供的外部工具 {remote_name}。")
        # MCP 用 inputSchema 描述入参；缺省时退回「无参」schema，保证 schema() 不崩
        schema = spec.get("inputSchema") or spec.get("input_schema") or {}
        self.parameters = (schema if isinstance(schema, dict) and schema
                           else {"type": "object", "properties": {}})

    async def run(self, **kwargs: Any) -> str:
        """透传到远端执行；把协议层错误转成可读文本（不让单个工具崩掉整轮）。"""
        try:
            return await self._client.call_tool(self._remote, kwargs)
        except MCPError as e:
            logger.warning("MCP 工具 %s 调用失败：%s", self.name, e)
            return f"MCP 工具调用失败：{e}"


class MCPManager:
    """按配置拉起若干 MCP 服务器，并把发现的工具汇入本机工具表。"""

    def __init__(self, servers: list[dict[str, Any]] | None = None, *,
                 timeout: float = 30.0, name_prefix: str = NAME_PREFIX) -> None:
        self._servers = list(servers or [])
        self._timeout = max(float(timeout), 1.0)
        self._prefix = name_prefix
        self._clients: list[MCPClient] = []
        self.tools: dict[str, MCPTool] = {}          # 已发现的 MCP 工具（name → Tool）

    # ------------------------------------------------------------------ 装配
    @classmethod
    def from_config(cls, cfg: Any) -> "MCPManager":
        """按 config.yaml 的 `mcp` 小节构造；总闸关闭或配置异常时返回空管理器。"""
        section = cfg.section("mcp") if cfg is not None else {}
        if not isinstance(section, dict) or not bool(section.get("enabled", False)):
            return cls([])
        servers = section.get("servers") or []
        if not isinstance(servers, list):
            logger.warning("mcp.servers 配置非法（应为列表），已忽略")
            servers = []
        try:
            timeout = float(section.get("timeout", 30))
        except (TypeError, ValueError):
            timeout = 30.0
        return cls([s for s in servers if isinstance(s, dict)], timeout=timeout)

    # ------------------------------------------------------------------ 连接
    async def connect_all(self) -> None:
        """逐个服务器：拉起 → 发现工具 → 包装。

        单个服务器失败只记日志并跳过，绝不因此中断其它服务器或主链。
        """
        for spec in self._servers:
            if not bool(spec.get("enabled", True)):
                continue
            name = str(spec.get("name") or "").strip()
            command = str(spec.get("command") or "").strip()
            if not name or not command:
                logger.warning("跳过配置不完整的 MCP 服务器（缺 name/command）：%r", spec)
                continue
            client = MCPClient(name, command,
                               args=spec.get("args"), env=spec.get("env"),
                               cwd=spec.get("cwd"), timeout=self._timeout)
            try:
                await client.start()
                specs = await client.list_tools()
            except (MCPError, OSError) as e:
                logger.warning("MCP 服务器 %s 接入失败：%s", name, e)
                await client.stop()
                continue
            self._clients.append(client)
            added = 0
            for tool_spec in specs:
                remote = str(tool_spec.get("name") or "").strip()
                if not remote:
                    continue
                wrapper = MCPTool(client, remote, tool_spec, name_prefix=self._prefix)
                if wrapper.name in self.tools:
                    logger.warning("MCP 工具名冲突，忽略：%s", wrapper.name)
                    continue
                self.tools[wrapper.name] = wrapper
                added += 1
            logger.info("MCP 服务器 %s 接入完成（%d 个工具）", name, added)

    def inject(self, tools: dict[str, Any]) -> int:
        """把已发现的工具合入装配层的 tools dict（同对象引用 → AgentLoop 立即可见）。

        Returns: 本次新合入的工具个数。
        """
        added = 0
        for name, tool in self.tools.items():
            if name in tools:
                continue
            tools[name] = tool
            added += 1
        return added

    # ------------------------------------------------------------------ 收尾
    async def stop(self) -> None:
        """关闭所有服务器连接（幂等）。"""
        for client in self._clients:
            try:
                await client.stop()
            except Exception:  # noqa: BLE001
                logger.exception("MCP 服务器 %s 关闭异常", client.name)
        self._clients.clear()
        self.tools.clear()
