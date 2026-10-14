"""JSON API 路由与鉴权（阶段二新增，见方案 §5.3）。

分层：
    server.py（HTTP 线程）→ api.py（路由/鉴权/入参）→ admin.service（业务）。

约定：
    * 统一 JSON 出入参；异常统一转成 {status, body} 形式的错误响应；
    * 鉴权：除 /api/setup、/api/login、/api/logout、/api/session、
      /static/*、/、/onebot/events 之外，全部要求有效会话 Cookie；
    * 角色：viewer 只读，admin 可写，owner 可管理后台账号与全局设置。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from .auth import role_at_least

logger = logging.getLogger(__name__)

# 静态资源根目录：ska/web/static/
STATIC_DIR = Path(__file__).resolve().parent / "static"
# 会话 Cookie 名
COOKIE_NAME = "ska_session"

# 静态资源的 MIME 映射（只列前端用到的几种）
_MIME = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml",
         ".png": "image/png", ".ico": "image/x-icon", ".json": "application/json"}


# ===========================================================================
# 请求 / 响应
# ===========================================================================
@dataclass
class Request:
    """一次 HTTP 请求（由 server.py 组装）。"""

    method: str
    path: str
    query: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)
    cookies: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    user: dict[str, str] | None = None       # 鉴权通过后填充 {username, role}

    def json(self) -> dict[str, Any]:
        """解析 JSON 请求体；空体或非法 JSON 返回 {}（写操作自行校验必填项）。"""
        if not self.body:
            return {}
        try:
            data = json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def q(self, key: str, default: Any = None) -> Any:
        return self.query.get(key, default)


@dataclass
class Response:
    """一次 HTTP 响应。"""

    status: int = 200
    body: bytes = b""
    content_type: str = "application/json; charset=utf-8"
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def json(cls, status: int, payload: Any, headers: dict[str, str] | None = None) -> "Response":
        text = json.dumps(payload, ensure_ascii=False, default=str)
        return cls(status=status, body=text.encode("utf-8"), headers=headers or {})

    @classmethod
    def ok(cls, payload: Any = None) -> "Response":
        return cls.json(200, {"ok": True, **({} if payload is None else payload)})

    @classmethod
    def error(cls, status: int, message: str) -> "Response":
        return cls.json(status, {"ok": False, "error": message})

    @classmethod
    def raw(cls, status: int, data: bytes, content_type: str) -> "Response":
        return cls(status=status, body=data, content_type=content_type)


# ===========================================================================
# 路由
# ===========================================================================
Handler = Callable[[Request, dict[str, str]], Awaitable[Response]]


@dataclass
class Route:
    method: str
    pattern: str
    handler: Handler
    min_role: str | None = None      # None = 公开（无需登录）


def _match(pattern: str, path: str) -> dict[str, str] | None:
    """按段匹配路由，支持 {name} 占位符；不匹配返回 None。"""
    p_segments = pattern.strip("/").split("/")
    path_segments = path.strip("/").split("/")
    if len(p_segments) != len(path_segments):
        return None
    params: dict[str, str] = {}
    for p, value in zip(p_segments, path_segments):
        if p.startswith("{") and p.endswith("}"):
            params[p[1:-1]] = value
        elif p != value:
            return None
    return params


class API:
    """路由表 + 鉴权 + 处理转发。"""

    def __init__(self, service: Any, auth: Any) -> None:
        self.service = service
        self.auth = auth
        self.routes: list[Route] = self._build_routes()

    # ------------------------------------------------------------------ 路由表
    def _build_routes(self) -> list[Route]:
        """集中声明所有路由（顺序敏感：具体路径要放在含 {} 的路径之前）。"""
        R = Route
        return [
            # ---- 公开：前端外壳与事件上报 ----
            R("GET", "/", self.h_index),
            R("GET", "/static/{path}", self.h_static),
            R("GET", "/api/session", self.h_session),
            R("POST", "/api/setup", self.h_setup),
            R("POST", "/api/login", self.h_login),
            R("POST", "/api/logout", self.h_logout),
            R("POST", "/onebot/events", self.h_onebot_events),

            # ---- 运行状态 ----
            R("GET", "/api/status", self.h_status, "viewer"),

            # ---- 1. 首次初始化（与 /api/setup 同义，供向导页查询状态）----
            R("GET", "/api/setup", self.h_setup_state),

            # ---- 2. 绑定 QQ 适配器 ----
            R("GET", "/api/qq/config", self.h_qq_get, "viewer"),
            R("PUT", "/api/qq/config", self.h_qq_put, "admin"),
            R("POST", "/api/qq/test", self.h_qq_test, "viewer"),
            R("GET", "/api/qq/login-info", self.h_qq_login_info, "viewer"),
            R("GET", "/api/qq/candidates", self.h_qq_candidates, "viewer"),

            # ---- 3. API 管理（提供商）----
            R("GET", "/api/providers", self.h_providers, "viewer"),
            R("POST", "/api/providers", self.h_provider_add, "admin"),
            R("POST", "/api/providers/{id}/activate", self.h_provider_activate, "admin"),
            R("POST", "/api/providers/{id}/test", self.h_provider_test, "viewer"),
            R("PUT", "/api/providers/{id}", self.h_provider_update, "admin"),
            R("DELETE", "/api/providers/{id}", self.h_provider_remove, "admin"),

            # ---- 4. 用量管理 ----
            R("GET", "/api/usage", self.h_usage, "viewer"),

            # ---- 5. 聊天用户管理 ----
            R("GET", "/api/users", self.h_users, "viewer"),
            R("PUT", "/api/users/{channel}/{user_id}", self.h_user_update, "admin"),

            # ---- 6. 记忆管理 ----
            R("GET", "/api/memory/long", self.h_mem_long, "viewer"),
            R("POST", "/api/memory/long", self.h_mem_long_add, "admin"),
            R("PUT", "/api/memory/long/{id}", self.h_mem_long_update, "admin"),
            R("DELETE", "/api/memory/long/{id}", self.h_mem_long_delete, "admin"),
            R("GET", "/api/memory/sessions", self.h_mem_sessions, "viewer"),
            R("DELETE", "/api/memory/sessions/{id}", self.h_mem_session_delete, "admin"),
            R("GET", "/api/memory/profile", self.h_mem_profiles, "viewer"),
            R("GET", "/api/memory/messages", self.h_mem_messages, "viewer"),
            R("DELETE", "/api/memory/messages/{id}", self.h_mem_message_delete, "admin"),

            # ---- 7. 权限管理（后台账号 + 等级规则）----
            R("GET", "/api/admins", self.h_admins, "owner"),
            R("POST", "/api/admins", self.h_admin_add, "owner"),
            R("PUT", "/api/admins/{id}", self.h_admin_update, "owner"),
            R("DELETE", "/api/admins/{id}", self.h_admin_delete, "owner"),
            R("GET", "/api/roles", self.h_roles, "viewer"),
            R("PUT", "/api/roles", self.h_roles_update, "owner"),

            # ---- 8. 插件拓展 ----
            R("GET", "/api/plugins", self.h_plugins, "viewer"),
            R("POST", "/api/plugins/{name}/toggle", self.h_plugin_toggle, "admin"),

            # ---- 9. 智能体权限（工具 + 全局开关）----
            R("GET", "/api/tools", self.h_tools, "viewer"),
            R("PUT", "/api/tools/{name}", self.h_tool_update, "admin"),
            R("GET", "/api/agent", self.h_agent_get, "viewer"),
            R("PUT", "/api/agent", self.h_agent_put, "admin"),

            # ---- 10. 可发布目标 ----
            R("GET", "/api/targets", self.h_targets, "viewer"),
            R("POST", "/api/targets", self.h_target_add, "admin"),
            R("PUT", "/api/targets/{id}", self.h_target_update, "admin"),
            R("DELETE", "/api/targets/{id}", self.h_target_delete, "admin"),

            # ---- 11. 设置 ----
            R("GET", "/api/settings", self.h_settings, "viewer"),
            R("PUT", "/api/settings", self.h_settings_update, "owner"),
        ]

    # ------------------------------------------------------------------ 分发入口
    async def handle(self, req: Request) -> Response:
        """匹配路由 → 鉴权 → 调用处理函数 → 统一异常兜底。"""
        for route in self.routes:
            if route.method != req.method:
                continue
            params = _match(route.pattern, req.path)
            if params is None:
                continue
            try:
                if not await self._authorize(req, route):
                    return (Response.error(401, "未登录或会话已过期")
                            if req.user is None
                            else Response.error(403, "权限不足"))
                result = route.handler(req, params)
                return await result if isinstance(result, Awaitable) else result
            except _APIError as e:                       # 业务侧主动抛出的可读错误
                return Response.error(e.status, e.message)
            except ValueError as e:
                return Response.error(400, str(e))
            except Exception:  # noqa: BLE001 —— 兜底，避免线程里抛异常
                logger.exception("API 处理失败：%s %s", req.method, req.path)
                return Response.error(500, "服务器内部错误")
        return Response.error(404, f"未找到接口：{req.method} {req.path}")

    async def _authorize(self, req: Request, route: Route) -> bool:
        """鉴权：公开路由直接放行；其余校验会话与角色（并填充 req.user）。"""
        if route.min_role is None:
            return True
        token = req.cookies.get(COOKIE_NAME, "")
        user = await self.auth.resolve(token)
        if user is None:
            req.user = None
            return False
        req.user = user
        return route.min_role is None or role_at_least(user["role"], route.min_role)

    # ================================================================== 公开路由
    async def h_index(self, req: Request, params: dict[str, str]) -> Response:
        """返回单页应用外壳。"""
        return self._serve_static("index.html")

    async def h_static(self, req: Request, params: dict[str, str]) -> Response:
        """返回静态资源（带目录穿越防护）。"""
        return self._serve_static(params["path"])

    def _serve_static(self, rel: str) -> Response:
        """安全读取静态文件：解析真实路径后必须仍在 STATIC_DIR 内。"""
        candidate = (STATIC_DIR / rel).resolve()
        try:
            candidate.relative_to(STATIC_DIR.resolve())      # 越界会抛 ValueError
        except ValueError:
            return Response.error(403, "非法路径")
        if not candidate.is_file():
            return Response.error(404, "资源不存在")
        mime = _MIME.get(candidate.suffix.lower(), "application/octet-stream")
        return Response.raw(200, candidate.read_bytes(), mime)

    async def h_session(self, req: Request, params: dict[str, str]) -> Response:
        """前端启动时调用：告知是否需要初始化、是否已登录、当前角色。"""
        needs_setup = await self.auth.needs_setup()
        user = await self.auth.resolve(req.cookies.get(COOKIE_NAME, "")) if not needs_setup else None
        return Response.ok({
            "needs_setup": needs_setup,
            "authenticated": user is not None,
            "username": (user or {}).get("username", ""),
            "role": (user or {}).get("role", ""),
        })

    async def h_setup_state(self, req: Request, params: dict[str, str]) -> Response:
        """初始化向导查询接口（GET /api/setup）。"""
        return Response.ok({"needs_setup": await self.auth.needs_setup()})

    async def h_setup(self, req: Request, params: dict[str, str]) -> Response:
        """首次初始化：创建管理员账号并直接登录。

        允许在同一请求里顺带保存初始提供商与 QQ 参数（都可留空跳过）。
        """
        data = req.json()
        username = str(data.get("username") or "admin").strip()
        password = str(data.get("password") or "")
        if not await self.auth.needs_setup():
            raise _APIError(409, "后台已初始化")
        if len(password) < 6:
            raise _APIError(400, "口令至少 6 位")
        result = await self.auth.setup(username, password)

        # 可选的初始设置（跳过也不影响）
        if isinstance(data.get("provider"), dict) and data["provider"].get("base_url"):
            self.service.providers.add(data["provider"])
        if isinstance(data.get("qq"), dict):
            await self.service.qq_update(data["qq"])
        if isinstance(data.get("app"), dict):
            await self.service.update_settings({"app": data["app"]})

        return Response.json(200, {"ok": True, **result},
                             headers={"Set-Cookie": self._cookie(result["token"])})

    async def h_login(self, req: Request, params: dict[str, str]) -> Response:
        data = req.json()
        result = await self.auth.login(str(data.get("username") or ""),
                                       str(data.get("password") or ""))
        if result is None:
            raise _APIError(401, "账号或口令不正确")
        return Response.json(200, {"ok": True, **result},
                             headers={"Set-Cookie": self._cookie(result["token"])})

    async def h_logout(self, req: Request, params: dict[str, str]) -> Response:
        await self.auth.logout(req.cookies.get(COOKIE_NAME, ""))
        return Response.json(200, {"ok": True},
                             headers={"Set-Cookie": self._cookie("", expire=True)})

    async def h_onebot_events(self, req: Request, params: dict[str, str]) -> Response:
        """SnowLuma httpClients 的事件上报入口。

        安全：用独立的 report_token 校验来源（可留空表示不校验，仅本机监听）。
        """
        expected = str(req.query.get("token") or self.service.settings.qq().get("report_token") or "")
        provided = str(req.query.get("access_token") or "")
        header_auth = req.headers.get("authorization", "")
        if header_auth.lower().startswith("bearer "):
            provided = header_auth[7:]
        if expected and provided != expected:
            raise _APIError(403, "上报令牌不正确")
        data = req.json()
        channel = self.service.app.get("channels", {}).get("qq")
        if channel is None:
            raise _APIError(503, "QQ 渠道未装配")
        accepted = await channel.on_event(data)
        return Response.json(200, {"ok": True, "accepted": accepted})

    @staticmethod
    def _cookie(token: str, expire: bool = False) -> str:
        """构造会话 Cookie（HttpOnly + SameSite=Strict，前端 JS 读不到）。"""
        if expire:
            return f"{COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"
        return f"{COOKIE_NAME}={token}; Path=/; HttpOnly; SameSite=Strict"

    # ================================================================== 状态
    async def h_status(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok(await self.service.status())

    # ================================================================== 提供商
    async def h_providers(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({
            "active": self.service.providers.active_id(),
            "providers": self.service.providers.public_list(),
            "presets": self.service.providers.presets(),
        })

    async def h_provider_add(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({"provider": self.service.providers.add(req.json())})

    async def h_provider_update(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({"provider": self.service.providers.update(params["id"], req.json())})

    async def h_provider_remove(self, req: Request, params: dict[str, str]) -> Response:
        self.service.providers.remove(params["id"])
        return Response.ok()

    async def h_provider_activate(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({"provider": self.service.providers.activate(params["id"])})

    async def h_provider_test(self, req: Request, params: dict[str, str]) -> Response:
        return Response.json(200, await self.service.providers.test(params["id"]))

    # ================================================================== 用量
    async def h_usage(self, req: Request, params: dict[str, str]) -> Response:
        days = _as_int(str(req.q("range", "7d")).rstrip("d"), 7)
        group_by = str(req.q("group", "provider"))
        if group_by not in ("provider", "model", "task", "day"):
            group_by = "provider"
        return Response.ok(await self.service.usage(days=days, group_by=group_by))

    # ================================================================== 聊天用户
    async def h_users(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({"users": await self.service.list_users()})

    async def h_user_update(self, req: Request, params: dict[str, str]) -> Response:
        row = await self.service.update_user(params["channel"], params["user_id"], req.json())
        return Response.ok({"user": row})

    # ================================================================== 记忆
    async def h_mem_long(self, req: Request, params: dict[str, str]) -> Response:
        rows = await self.service.memory_long(
            limit=_as_int(str(req.q("limit", "100")), 100),
            offset=_as_int(str(req.q("offset", "0")), 0),
            user=str(req.q("user", "")))
        return Response.ok({"items": rows})

    async def h_mem_long_add(self, req: Request, params: dict[str, str]) -> Response:
        data = req.json()
        content = str(data.get("content") or "").strip()
        if not content:
            raise _APIError(400, "content 不能为空")
        created = await self.service.memory_add(
            content, importance=_as_float(data.get("importance"), 0.5),
            user=str(data.get("user") or ""))
        return Response.ok(created)

    async def h_mem_long_update(self, req: Request, params: dict[str, str]) -> Response:
        ok = await self.service.memory_update(int(params["id"]), req.json())
        return Response.ok({"updated": ok})

    async def h_mem_long_delete(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({"deleted": await self.service.memory_delete(int(params["id"]))})

    async def h_mem_sessions(self, req: Request, params: dict[str, str]) -> Response:
        rows = await self.service.memory_sessions(
            limit=_as_int(str(req.q("limit", "50")), 50))
        return Response.ok({"items": rows})

    async def h_mem_session_delete(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({"deleted": await self.service.memory_session_delete(int(params["id"]))})

    async def h_mem_profiles(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({"items": await self.service.memory_profiles()})

    async def h_mem_messages(self, req: Request, params: dict[str, str]) -> Response:
        rows = await self.service.memory_messages(
            limit=_as_int(str(req.q("limit", "100")), 100), user=str(req.q("user", "")))
        return Response.ok({"items": rows})

    async def h_mem_message_delete(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({"deleted": await self.service.memory_message_delete(int(params["id"]))})

    # ================================================================== 后台账号 / 角色
    async def h_admins(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({"admins": await self.service.store.list_admins()})

    async def h_admin_add(self, req: Request, params: dict[str, str]) -> Response:
        data = req.json()
        username = str(data.get("username") or "").strip()
        password = str(data.get("password") or "")
        role = str(data.get("role") or "admin")
        if not username or len(password) < 6:
            raise _APIError(400, "用户名必填且口令至少 6 位")
        from .auth import hash_password
        pass_hash, salt = hash_password(password)
        admin_id = await self.service.store.create_admin(username, pass_hash, salt, role)
        return Response.ok({"id": admin_id})

    async def h_admin_update(self, req: Request, params: dict[str, str]) -> Response:
        from .auth import hash_password
        data = req.json()
        pass_hash = salt = None
        if data.get("password"):
            pass_hash, salt = hash_password(str(data["password"]))
        ok = await self.service.store.update_admin(
            int(params["id"]), role=data.get("role"), pass_hash=pass_hash, salt=salt)
        return Response.ok({"updated": ok})

    async def h_admin_delete(self, req: Request, params: dict[str, str]) -> Response:
        admins = await self.service.store.list_admins()
        if len(admins) <= 1:
            raise _APIError(400, "至少保留一个后台账号")
        return Response.ok({"deleted": await self.service.store.delete_admin(int(params["id"]))})

    async def h_roles(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok(await self.service.roles_get())

    async def h_roles_update(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok(await self.service.roles_update(req.json()))

    # ================================================================== 插件
    async def h_plugins(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({"plugins": await self.service.list_plugins()})

    async def h_plugin_toggle(self, req: Request, params: dict[str, str]) -> Response:
        enabled = bool(req.json().get("enabled", True))
        return Response.ok(await self.service.toggle_plugin(params["name"], enabled))

    # ================================================================== 工具 / 智能体
    async def h_tools(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({"tools": await self.service.list_tools()})

    async def h_tool_update(self, req: Request, params: dict[str, str]) -> Response:
        data = req.json()
        return Response.ok(await self.service.update_tool(
            params["name"], enabled=bool(data.get("enabled", True)),
            min_level=str(data.get("min_level") or "member"),
            note=str(data.get("note") or "")))

    async def h_agent_get(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok(await self.service.agent_get())

    async def h_agent_put(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok(await self.service.agent_update(req.json()))

    # ================================================================== 可发布目标
    async def h_targets(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({"targets": await self.service.list_targets()})

    async def h_target_add(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok(await self.service.add_target(req.json()))

    async def h_target_update(self, req: Request, params: dict[str, str]) -> Response:
        ok = await self.service.update_target(int(params["id"]), req.json())
        return Response.ok({"updated": ok})

    async def h_target_delete(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok({"deleted": await self.service.delete_target(int(params["id"]))})

    # ================================================================== 设置
    async def h_settings(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok(await self.service.settings_snapshot())

    async def h_settings_update(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok(await self.service.update_settings(req.json()))

    # ================================================================== QQ
    async def h_qq_get(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok(await self.service.qq_config())

    async def h_qq_put(self, req: Request, params: dict[str, str]) -> Response:
        return Response.ok(await self.service.qq_update(req.json()))

    async def h_qq_test(self, req: Request, params: dict[str, str]) -> Response:
        return Response.json(200, await self.service.qq_test(req.json() or None))

    async def h_qq_login_info(self, req: Request, params: dict[str, str]) -> Response:
        return Response.json(200, await self.service.qq_login_info())

    async def h_qq_candidates(self, req: Request, params: dict[str, str]) -> Response:
        return Response.json(200, await self.service.qq_candidates())


class _APIError(Exception):
    """带 HTTP 状态码的业务错误（由 handle 统一转成 JSON 错误响应）。"""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------- 小工具
def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default