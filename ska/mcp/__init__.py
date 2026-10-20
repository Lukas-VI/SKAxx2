"""MCP 接入包：用标准库实现的极简 MCP（Model Context Protocol）客户端。

为什么自研而不装官方 SDK：
    requirements.txt 的硬约束是「内核零重依赖」。官方 `mcp` SDK 会拖入
    pydantic-settings / anyio / httpx-sse 等一整套，与「薄内核」取向不符。
    智能体真正需要的只有三件事——握手、发现工具、调用工具，标准库
    （asyncio.subprocess + json）足以实现。

模块分工：
    * client.py —— MCPClient：stdio 上的 JSON-RPC 2.0 客户端（进程 + 收发配对）；
    * bridge.py —— MCPTool / MCPManager：把外部工具包装成本机 Tool 并合入工具表。

启用方式见 config.yaml 的 `mcp` 小节：默认关闭，填好 servers 后打开总闸即可。
"""
from .bridge import MCPManager, MCPTool
from .client import MCPClient, MCPError, extract_text

__all__ = ["MCPClient", "MCPError", "extract_text", "MCPTool", "MCPManager"]
