"""内置工具：time / calculator / fetch / memory_ops。

这些是第一期就能跑起来的「自带能力」，用于替代老项目依赖 qwen_agent + 7 个
MCP 子进程的方案（启动慢、无生命周期管理）。第三期如需外部工具，再按官方 mcp SDK 接。
"""
from __future__ import annotations

import ast
import datetime as _dt
import logging
import math
import operator as _op
import re
from typing import Any

import httpx

from .base import Tool, decode_bytes, tool

logger = logging.getLogger(__name__)

# 工具返回统一截断上限：防止把超长网页塞进上下文造成 prompt 注入/爆窗
MAX_OBSERVATION = 2000


def _clip(text: str, limit: int = MAX_OBSERVATION) -> str:
    """截断过长文本。"""
    text = text or ""
    return text if len(text) <= limit else text[:limit] + f"\n…（已截断，共 {len(text)} 字）"


# ===========================================================================
# time —— 第一期验证用最小工具
# ===========================================================================
@tool("time")
class TimeTool(Tool):
    description = "获取当前本地日期与时间。当你需要知道现在几点、今天几号时调用。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "fmt": {
                "type": "string",
                "description": "strftime 格式串，默认 '%Y-%m-%d %H:%M:%S %A'",
            }
        },
    }

    async def run(self, fmt: str | None = None, **_: Any) -> str:
        now = _dt.datetime.now()
        return now.strftime(fmt or "%Y-%m-%d %H:%M:%S %A")


# ===========================================================================
# calculator —— 安全四则运算（AST 求值，不用 eval）
# ===========================================================================
# 允许的二元运算符
_BIN_OPS = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
    ast.Div: _op.truediv, ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod, ast.Pow: _op.pow,
}
# 允许的一元运算符
_UNARY_OPS = {ast.UAdd: _op.pos, ast.USub: _op.neg}
# 允许的函数名 → 实现（白名单，杜绝任意代码执行）
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max,
          "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil}


@tool("calculator")
class CalculatorTool(Tool):
    description = "计算数学表达式（支持 + - * / // % ** 与 sqrt/abs/round 等函数）。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "要计算的表达式，如 (3+4)*2"},
        },
        "required": ["expression"],
    }

    async def run(self, expression: str = "", **_: Any) -> str:
        if not expression:
            return "错误：表达式为空"
        try:
            value = self._eval(ast.parse(expression, mode="eval").body)
        except Exception as e:  # noqa: BLE001 —— 工具层把任何错误转成可读文本返回
            return f"计算失败：{e}"
        return str(value)

    def _eval(self, node: ast.AST) -> float:
        """递归求值，只放行白名单节点。"""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("不支持的常量类型")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"不支持的函数：{node.func.id}")
            return fn(*(self._eval(a) for a in node.args))
        raise ValueError("表达式包含不被允许的语法")


# ===========================================================================
# fetch —— 抓取网页正文（httpx，串行安全、截断防注入）
# ===========================================================================
_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_HTML_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")
# 网页自述编码：<meta charset="gbk"> 与 <meta http-equiv=... content="...; charset=gbk">
_META_CHARSET_RE = re.compile(rb"""charset\s*=\s*["']?\s*([A-Za-z0-9_\-]+)""", re.I)
_HEADER_CHARSET_RE = re.compile(r"charset\s*=\s*([\w\-]+)", re.I)


def _sniff_charset(raw: bytes, content_type: str = "") -> str:
    """猜网页编码：优先页面内 <meta charset>，其次响应头 Content-Type，猜不到返回空串。

    为什么 meta 优先：不少中文站点响应头只写 `text/html`（或错误的 iso-8859-1），
    但页面里老老实实写了 `charset=gbk`——只信响应头会把整页中文解成乱码。
    """
    head = raw[:4096].lower()                      # 只看前 4KB，够定位 <head> 里的声明
    match = _META_CHARSET_RE.search(head)
    if match:
        declared = match.group(1).decode("ascii", "ignore")
    else:
        header = _HEADER_CHARSET_RE.search(content_type or "")
        declared = header.group(1) if header else ""
    # 单字节的「服务器默认值」基本不可信：大量 UTF-8 页面被服务器硬标成
    # iso-8859-1，按它解码虽然不报错，但中文会整页变乱码。这类声明直接丢弃，
    # 交给 decode_bytes 的 UTF-8 → GB18030 自动探测。
    if declared.lower() in ("iso-8859-1", "latin-1", "us-ascii", "ascii"):
        return ""
    return declared


@tool("fetch")
class FetchTool(Tool):
    description = "抓取一个网页并返回其纯文本内容（已去除标签，最多 2000 字）。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的 http/https 地址"},
        },
        "required": ["url"],
    }
    timeout = 15.0

    async def run(self, url: str = "", **_: Any) -> str:
        if not url.startswith(("http://", "https://")):
            return "错误：只支持 http/https 地址"
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "SKAxx2/0.1"})
                resp.raise_for_status()
                # 不用 resp.text：httpx 在响应头没写 charset 时默认 UTF-8，
                # 抓 GBK 页面会整页乱码。这里先按 meta/响应头猜测，再交给 decode_bytes 兜底。
                raw = resp.content
                html = decode_bytes(raw, preferred=_sniff_charset(
                    raw, resp.headers.get("content-type", "")))
        except httpx.HTTPError as e:
            return f"抓取失败：{e}"

        # 去脚本/样式 → 去标签 → 压缩空行
        text = _TAG_RE.sub(" ", html)
        text = _HTML_RE.sub(" ", text)
        text = _WS_RE.sub("\n\n", text).strip()
        # 关键：工具结果截断后再回填上下文，降低 prompt 注入面
        return _clip(text)


# ===========================================================================
# memory_ops —— 记忆读写（第三期，依赖 MemoryStore）
# ===========================================================================
@tool("memory_ops")
class MemoryOpsTool(Tool):
    description = ("读写记忆。action=search 检索旧事；action=remember 记下一件重要的事；"
                   "action=pin 把一条内容提升为「核心记忆」（此后始终记得）；action=unpin 取消核心记忆。")
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["search", "remember", "pin", "unpin"]},
            "content": {"type": "string",
                        "description": "search/pin/unpin 的内容或关键词；remember 要记的内容"},
            "importance": {"type": "number",
                           "description": "remember/pin 时的重要性 0~1（pin 默认按 0.8 处理）"},
        },
        "required": ["action", "content"],
    }

    async def run(self, action: str = "", content: str = "", importance: float = 0.6,
                  **_: Any) -> str:
        store = getattr(self.deps, "store", None)
        if store is None:
            return "记忆库尚未启用。"
        if action == "search":
            hits = await store.search_long(content, top_k=5)
            return "；".join(hits) if hits else "没有想起相关的旧事。"
        if action == "remember":
            await store.add_long(content, source="manual", importance=float(importance))
            return "记住了。"
        if action == "pin":
            # 提升为核心记忆：始终注入决策提示词，不依赖关键词检索
            await store.pin_core(content, importance=max(float(importance), 0.8))
            return "已记进核心记忆，往后会一直记得。"
        if action == "unpin":
            removed = await store.unpin_core(content)
            return "已从核心记忆移除。" if removed else "核心记忆里没有这条内容。"
        return f"未知操作：{action}"