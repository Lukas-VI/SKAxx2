"""后台鉴权：口令哈希、会话令牌、首次初始化判定（决策 D2）。

安全设计（本地单管理员场景，够用且零依赖）：
    * 口令用 PBKDF2-HMAC-SHA256 + 随机 salt，迭代 10 万次，只存哈希不存明文；
    * 登录成功签发随机会话令牌（secrets.token_urlsafe(32)），落 sessions 表并带过期时间；
    * 令牌经 Cookie 下发（HttpOnly + SameSite=Strict），前端 JS 读不到，降低 XSS 风险；
    * 服务只监听 127.0.0.1（见 server.py），不暴露到局域网。
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time

logger = logging.getLogger(__name__)

# PBKDF2 迭代次数：10 万（本地登录一次约几十毫秒，可接受）
PBKDF2_ROUNDS = 100_000
PBKDF2_ALGO = "sha256"
SALT_BYTES = 16
TOKEN_BYTES = 32

# 后台账号角色（权限从高到低）——与 admin/store.py 的 role 字段取值一致
ROLES = ("owner", "admin", "viewer")
_ROLE_RANK = {"owner": 3, "admin": 2, "viewer": 1}


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """把口令哈希成 (hash_hex, salt_hex)；未给 salt 时随机生成。

    Returns:
        (pass_hash, salt)：两者都是十六进制字符串，直接落库。
    """
    salt_hex = salt or secrets.token_hex(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(PBKDF2_ALGO, password.encode("utf-8"),
                                 bytes.fromhex(salt_hex), PBKDF2_ROUNDS)
    return digest.hex(), salt_hex


def verify_password(password: str, pass_hash: str, salt: str) -> bool:
    """校验口令是否匹配（用 compare_digest 防时序侧信道）。"""
    if not pass_hash or not salt:
        return False
    try:
        expected, _ = hash_password(password, salt)
    except ValueError:                      # salt 不是合法十六进制
        return False
    return hmac.compare_digest(expected, pass_hash)


def new_token() -> str:
    """生成一个随机会话令牌。"""
    return secrets.token_urlsafe(TOKEN_BYTES)


def role_at_least(role: str, minimum: str) -> bool:
    """判断角色是否达到最低要求（owner > admin > viewer）。"""
    return _ROLE_RANK.get(role, 0) >= _ROLE_RANK.get(minimum, 0)


class AuthService:
    """会话与账号的业务封装（基于 AdminStore）。"""

    def __init__(self, store, session_hours: int = 72) -> None:
        self.store = store
        self.session_hours = max(int(session_hours), 1)

    # ------------------------------------------------------------------ 初始化
    async def needs_setup(self) -> bool:
        """是否处于「首次使用」状态（没有任何后台账号）。"""
        return await self.store.count_admins() == 0

    async def setup(self, username: str, password: str,
                    role: str = "owner") -> dict[str, str]:
        """首次初始化：创建管理员账号并直接签发一个会话（省去再登录一次）。

        Raises:
            RuntimeError: 已经初始化过（防止被调用方绕过 needs_setup 重复建号）。
        """
        if not await self.needs_setup():
            raise RuntimeError("后台已初始化，无法重复初始化")
        username = (username or "").strip() or "admin"
        if len(password or "") < 6:
            raise ValueError("口令至少 6 位")
        pass_hash, salt = hash_password(password)
        await self.store.create_admin(username, pass_hash, salt, role)
        logger.info("后台初始化完成，管理员账号：%s", username)
        token = await self._issue(username)
        return {"token": token, "username": username, "role": role}

    # ------------------------------------------------------------------ 登录 / 登出
    async def login(self, username: str, password: str) -> dict[str, str] | None:
        """校验账号口令；成功返回会话信息，失败返回 None。"""
        row = await self.store.get_admin(username or "")
        if row is None or not verify_password(password or "", row["pass_hash"], row["salt"]):
            return None
        await self.store.touch_login(username)
        token = await self._issue(username)
        return {"token": token, "username": username, "role": row["role"]}

    async def logout(self, token: str) -> None:
        if token:
            await self.store.delete_session(token)

    async def resolve(self, token: str) -> dict[str, str] | None:
        """用会话令牌换取身份信息（附带最新角色，角色变更即时生效）。"""
        if not token:
            return None
        session = await self.store.get_session(token)
        if session is None:
            return None
        row = await self.store.get_admin(session["username"])
        if row is None:                         # 账号已被删除 → 会话作废
            await self.store.delete_session(token)
            return None
        return {"username": row["username"], "role": row["role"]}

    # ------------------------------------------------------------------ 内部
    async def _issue(self, username: str) -> str:
        token = new_token()
        await self.store.create_session(token, username,
                                        time.time() + self.session_hours * 3600)
        return token