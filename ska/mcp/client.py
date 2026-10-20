"""极简 MCP 客户端：JSON-RPC 2.0 over stdio（标准库实现，无第三方依赖）。

协议要点（借鉴官方 MCP 实现与社区文档，只保留智能体必需的子集）：
    * 传输：子进程的 stdin/stdout 上「一行一个 JSON」的 JSON-RPC 2.0 报文；
    * 握手：先发 `initialize` 请求 → 拿到结果后再发 `notifications/initialized` 通知；
    * 配对：请求带自增 id，响应按同 id 回来；通知没有 id，不等回复；
    * 排障：服务器的日志通常写 stderr，必须单独消费，否则管道写满会把整个
      服务器卡死（这是各语言 MCP 客户端都踩过的坑）。

线程/并发模型：单个服务器一个子进程 + 一个「读 stdout」任务；所有请求共用该任务，
靠 id → Future 的字典把响应分发给各自的等待者。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# 客户端声明的协议版本与身份（服务器据此决定行为兼容性）。
PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO: dict[str, str] = {"name": "ska", "version": "0.1"}


class MCPError(RuntimeError):
    """MCP 协议层错误：握手失败 / 请求超时 / 服务器返回 error。"""


def extract_text(result: Any) -> str:
    """从 `tools/call` 的返回里抽出可读文本。

    MCP 的返回形如 `{"content": [{"type": "text", "text": "..."}], "isError": false}`，
    content 是分片数组，可能混合文本、图片、资源等类型；这里把文本类分片拼起来，
    非文本分片退化成其 JSON 文本，保证不丢信息也不崩。
    """
    if not isinstance(result, dict):
        return str(result)
    parts: list[str] = []
    for item in result.get("content") or []:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append(str(item.get("text") or ""))
        elif kind == "resource":
            resource = item.get("resource") or {}
            parts.append(str(resource.get("text") or resource.get("uri") or ""))
        else:
            parts.append(json.dumps(item, ensure_ascii=False))
    text = "\n".join(p for p in parts if p).strip()
    if not text:
        text = "（服务器返回了空结果）"
    return ("[MCP 返回错误] " + text) if result.get("isError") else text


class MCPClient:
    """一个外部 MCP 服务器连接（一个子进程）。"""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.name = name                      # 服务器名（工具名前缀用）
        self.command = command               # 可执行文件（如 npx / python / uvx）
        self.args = list(args or [])
        self.env = dict(env or {})           # 追加/覆盖的环境变量
        self.cwd = cwd or None
        self.timeout = max(float(timeout), 1.0)

        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._stderr_reader: asyncio.Task[None] | None = None
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._seq = 0                        # 请求 id 自增计数
        self._closed = False

    # ------------------------------------------------------------------ 生命周期
    async def start(self) -> "MCPClient":
        """拉起子进程并完成 initialize 握手；失败会清理并抛 MCPError。"""
        # 在继承的父进程环境上叠加服务器自定义 env（MCP 服务器常需要自己的 key 才能跑）
        env = dict(os.environ)
        env.update(self.env)
        env.setdefault("PYTHONUNBUFFERED", "1")   # 让子进程 stdout 即时刷新（JSON 行不粘包）
        try:
            self._proc = await asyncio.create_subprocess_exec(
                self.command, *self.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env, cwd=self.cwd,
            )
        except OSError as e:
            raise MCPError(f"无法启动 MCP 服务器 {self.name}（{self.command}）：{e}") from e

        self._reader = asyncio.create_task(self._read_stdout())
        self._stderr_reader = asyncio.create_task(self._drain_stderr())
        try:
            await self._initialize()
        except Exception:
            await self.stop()                 # 握手失败不留半死进程
            raise
        return self

    async def stop(self) -> None:
        """关闭连接并回收子进程（幂等，可重复调用）。"""
        self._closed = True
        self._fail_all("MCP 服务器正在关闭")
        tasks = [t for t in (self._reader, self._stderr_reader) if t is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)   # 等取消落地，避免告警
        self._reader = self._stderr_reader = None

        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None and not proc.stdin.is_closing():
                proc.stdin.close()            # 关 stdin 是让服务器优雅退出的首选方式
        except Exception:  # noqa: BLE001
            pass
        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), 5)
            except asyncio.TimeoutError:
                try:
                    proc.kill()               # 磨到最后才强杀
                except ProcessLookupError:
                    pass

    # ------------------------------------------------------------------ 协议方法
    async def list_tools(self) -> list[dict[str, Any]]:
        """发现服务器提供的工具列表（每项含 name / description / inputSchema）。"""
        result = await self._request("tools/list", {})
        tools = result.get("tools") if isinstance(result, dict) else None
        return [t for t in (tools or []) if isinstance(t, dict)]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """调用服务器上的一个工具，返回可读文本结果。"""
        result = await self._request("tools/call",
                                     {"name": name, "arguments": arguments or {}})
        return extract_text(result)

    # ------------------------------------------------------------------ 内部：握手
    async def _initialize(self) -> None:
        await self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        })
        await self._notify("notifications/initialized", {})
        logger.info("MCP 服务器 %s 握手完成", self.name)

    # ------------------------------------------------------------------ 内部：收发
    async def _send(self, payload: dict[str, Any]) -> None:
        """把一条 JSON-RPC 报文写进子进程 stdin（一行一个 JSON + 换行）。"""
        proc = self._proc
        if self._closed or proc is None or proc.stdin is None:
            raise MCPError(f"MCP 服务器 {self.name} 未运行")
        data = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        try:
            proc.stdin.write(data)
            await proc.stdin.drain()
        except (ConnectionResetError, BrokenPipeError, RuntimeError) as e:
            raise MCPError(f"写入 MCP 服务器 {self.name} 失败：{e}") from e

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """发一个带 id 的请求并等待其响应（超时抛 MCPError）。"""
        proc = self._proc
        if self._closed or proc is None or proc.returncode is not None:
            raise MCPError(f"MCP 服务器 {self.name} 未运行")
        self._seq += 1
        rid = self._seq
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        try:
            return await asyncio.wait_for(fut, self.timeout)
        except asyncio.TimeoutError as e:
            raise MCPError(f"MCP 请求 {method} 超时（>{self.timeout:.0f}s）") from e
        finally:
            self._pending.pop(rid, None)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """发一个没有 id 的通知（不需要也不等待响应）。"""
        await self._send({"jsonrpc": "2.0", "method": method, "params": params})

    async def _read_stdout(self) -> None:
        """后台任务：逐行读 stdout → 解析 JSON → 按 id 唤醒等待者。"""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        while True:
            try:
                line = await proc.stdout.readline()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.debug("MCP %s 读 stdout 异常：%s", self.name, e)
                break
            if not line:                       # 空行 = 进程已结束
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                logger.debug("MCP %s 输出非 JSON（已忽略）：%s", self.name, text[:200])
                continue
            self._dispatch(msg)
        # 进程结束：让所有挂起请求失败，避免调用方永久等待
        self._fail_all(f"MCP 服务器 {self.name} 连接已关闭")

    async def _drain_stderr(self) -> None:
        """后台任务：持续消费 stderr（否则管道写满会卡死服务器），只进调试日志。"""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while True:
            try:
                line = await proc.stderr.readline()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                break
            if not line:
                break
            logger.debug("[mcp:%s] %s", self.name, line.decode("utf-8", "replace").rstrip())

    def _dispatch(self, msg: Any) -> None:
        """把一条服务器报文分发给对应等待者（无 id 的通知直接忽略）。"""
        if not isinstance(msg, dict):
            return
        rid = msg.get("id")
        if rid is None:
            return                             # 服务器通知：本客户端暂无订阅需求
        fut = self._pending.pop(rid, None)
        if fut is None or fut.done():
            return
        if "error" in msg:
            err = msg.get("error") or {}
            message = err.get("message") if isinstance(err, dict) else str(err)
            fut.set_exception(MCPError(f"MCP 服务器 {self.name} 返回错误：{message}"))
        else:
            result = msg.get("result")
            fut.set_result(result if isinstance(result, dict) else {"value": result})

    def _fail_all(self, reason: str) -> None:
        """让所有挂起请求以异常结束（进程退出/关闭时调用）。"""
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(MCPError(reason))
        self._pending.clear()
