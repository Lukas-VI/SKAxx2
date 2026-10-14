"""WebUI HTTP 服务：标准库线程服务器 + 线程↔asyncio 桥（决策 D1）。

为什么这么设计：
    * 项目核心是 asyncio 事件循环（心跳/总线/LLM 都是异步的），
      而标准库 http.server 是同步阻塞模型；
    * 因此让 HTTPServer 跑在**守护线程**里，每个请求把协程经
      asyncio.run_coroutine_threadsafe 提交回主事件循环执行，再取回结果。
      这样所有业务逻辑（含 SQLite/httpx）都只在事件循环里跑，
      不存在「两个线程同时碰同一个连接」的竞态。

风险与对策（方案 §9）：
    * 线程↔循环桥写错会死锁 → 桥只在这里实现一次（run_coro），HTTP 线程只等结果；
    * 静态资源目录穿越 → 解析后校验是否仍在 static 目录前缀内；
    * 退出时端口未释放 → shutdown() + server_close()。
"""
from __future__ import annotations

import asyncio
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from .api import API, Request, Response

logger = logging.getLogger(__name__)


class _Handler(BaseHTTPRequestHandler):
    """把 HTTP 请求翻译成 api.Request，并把 api.Response 写回。"""

    # 引用注入：由 make_handler 动态挂到子类上（避免全局变量）
    api: API
    server_ref: "WebServer"

    # ------------------------------------------------------------------ 路由分发
    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        """组装请求 → 交给 API（在事件循环里执行）→ 写响应。"""
        try:
            req = self._build_request(method)
            resp = self.server_ref.run_coro(self.api.handle(req))
        except Exception:  # noqa: BLE001 —— HTTP 层兜底，绝不把异常抛到线程外
            logger.exception("处理请求失败：%s %s", method, self.path)
            resp = Response.json(500, {"error": "服务器内部错误"})
        self._write(resp)

    def _build_request(self, method: str) -> Request:
        """读取路径/查询串/请求头/Cookie/请求体，打包成 Request。"""
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length > 0 else b""
        return Request(
            method=method,
            path=path,
            query={k: (v[0] if len(v) == 1 else v) for k, v in query.items()},
            headers={k.lower(): v for k, v in self.headers.items()},
            cookies=_parse_cookie(self.headers.get("Cookie", "")),
            body=body,
        )

    def _write(self, resp: Response) -> None:
        """把 Response 写成 HTTP 报文（含自定义头/Cookie）。"""
        try:
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.content_type)
            self.send_header("Content-Length", str(len(resp.body)))
            # 后台是本地单页应用，禁用缓存避免改前端后看不到效果
            self.send_header("Cache-Control", "no-store")
            for key, value in resp.headers.items():
                self.send_header(key, value)
            self.end_headers()
            if resp.body:
                self.wfile.write(resp.body)
        except (BrokenPipeError, ConnectionResetError):   # 浏览器提前断开属正常
            logger.debug("客户端提前断开连接")

    # 默认会把每个请求打到 stderr；这里降到 debug，保持控制台清爽
    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug("[web] " + fmt, *args)


def _parse_cookie(header: str) -> dict[str, str]:
    """把 Cookie 头解析成 dict（只做最简 k=v; 切分，够用）。"""
    out: dict[str, str] = {}
    for part in (header or "").split(";"):
        if "=" in part:
            key, _, value = part.partition("=")
            out[key.strip()] = value.strip()
    return out


class WebServer:
    """后台 HTTP 服务的生命周期管理。"""

    def __init__(self, api: API, host: str = "127.0.0.1", port: int = 5098) -> None:
        self.api = api
        self.host = host
        self.port = int(port)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ 生命周期
    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """在后台线程启动 HTTP 服务（必须在事件循环内调用，用于捕获 loop）。"""
        self._loop = loop
        handler = type("_BoundHandler", (_Handler,),
                       {"api": self.api, "server_ref": self})
        # 端口占用等错误在此抛出（main.py 会捕获并提示改端口）
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="ska-web", daemon=True)
        self._thread.start()
        logger.info("WebUI 已启动：http://%s:%d", self.host, self.port)

    def stop(self) -> None:
        """停止 HTTP 服务并释放端口（幂等）。"""
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:  # noqa: BLE001
                logger.debug("关闭 WebUI 服务异常", exc_info=True)
            self._httpd = None
        logger.info("WebUI 已停止")

    # ------------------------------------------------------------------ 线程 ↔ 循环桥
    def run_coro(self, coro, timeout: float = 60.0):
        """把协程提交到主事件循环执行并等待结果（HTTP 线程调用入口）。

        Args:
            coro:    待执行的协程对象。
            timeout: 等待上限，超时抛 TimeoutError（避免请求线程永久卡死）。

        Raises:
            RuntimeError: 事件循环尚未捕获（服务未启动）。
        """
        if self._loop is None:
            raise RuntimeError("WebServer 尚未启动（未绑定事件循环）")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)