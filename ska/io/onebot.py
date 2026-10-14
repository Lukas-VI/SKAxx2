"""OneBot v11 HTTP 客户端（SnowLuma 适配器薄封装，阶段二新增）。

事实依据（见方案 §3 的查证结论）：
    * SnowLuma 是 OneBot v11 兼容实现，action 通过 **HTTP POST** 调用，
      地址形如 `{api_base}/{action}`（默认 `http://127.0.0.1:3000/send_group_msg`）；
    * 鉴权用 accessToken，支持 `Authorization: Bearer <token>`；
    * 响应统一是 OneBot envelope：`{status, retcode, data, echo}`，
      retcode != 0（或 status == "failed"）即视为失败。

本模块只做「HTTP 调用 + 响应解包」，不含任何业务判断，
这样 qq.py（渠道）与后台「绑定 QQ」页都能复用它。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class OneBotError(RuntimeError):
    """OneBot 调用失败（网络/超时/协议错误/retcode 非 0）。"""


def text_message(content: str) -> list[dict[str, Any]]:
    """把纯文本包装成 OneBot 消息段数组（array 格式，SnowLuma 默认）。"""
    return [{"type": "text", "data": {"text": content}}]


class OneBotClient:
    """OneBot v11 HTTP API 客户端。

    用法：
        client = OneBotClient("http://127.0.0.1:3000", access_token="xxx")
        info = await client.get_login_info()
        await client.send_group_msg(965244857, "晚上好")
        await client.aclose()
    """

    def __init__(self, api_base: str, access_token: str = "", timeout: float = 30.0,
                 *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        # api_base 形如 http://127.0.0.1:3000（去掉结尾斜杠，避免拼出双斜杠）
        self.api_base = str(api_base or "").rstrip("/")
        self.access_token = str(access_token or "")
        self.timeout = float(timeout)
        self._transport = transport          # 仅供测试注入 MockTransport
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------ 生命周期
    def _ensure_client(self) -> httpx.AsyncClient:
        """惰性创建并复用 AsyncClient（带鉴权头）。"""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.access_token:            # 无 token 时不发空的 Authorization 头
                headers["Authorization"] = f"Bearer {self.access_token}"
            self._client = httpx.AsyncClient(headers=headers,
                                             timeout=httpx.Timeout(self.timeout + 5.0),
                                             transport=self._transport)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ 核心调用
    async def call(self, action: str, **params: Any) -> Any:
        """调用一个 OneBot action，返回解包后的 data。

        Args:
            action: action 名（如 send_group_msg）。
            params: 该 action 的参数（值为 None 的键会被剔除）。

        Raises:
            OneBotError: 网络/超时/JSON 解析失败，或 retcode 非 0。
        """
        if not self.api_base:
            raise OneBotError("未配置 OneBot api_base（请在后台「绑定 QQ」页填写）")
        url = f"{self.api_base}/{action}"
        payload = {k: v for k, v in params.items() if v is not None}
        client = self._ensure_client()
        try:
            async with asyncio.timeout(self.timeout):
                resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except asyncio.TimeoutError as e:
            raise OneBotError(f"OneBot 调用超时（>{self.timeout}s）：{action}") from e
        except httpx.HTTPStatusError as e:
            raise OneBotError(
                f"OneBot 返回 {e.response.status_code}：{e.response.text[:200]}") from e
        except httpx.HTTPError as e:
            raise OneBotError(f"OneBot 网络错误：{e}") from e
        except ValueError as e:              # resp.json() 解析失败
            raise OneBotError(f"OneBot 响应不是合法 JSON：{action}") from e

        return self._unwrap(action, data)

    @staticmethod
    def _unwrap(action: str, data: Any) -> Any:
        """解包 OneBot envelope，失败即抛错。"""
        if not isinstance(data, dict):
            return data                       # 某些实现可能直返 data
        retcode = data.get("retcode")
        status = str(data.get("status", "")).lower()
        if status == "failed" or (retcode is not None and int(retcode) != 0):
            raise OneBotError(f"OneBot 执行失败 {action}：status={status} "
                              f"retcode={retcode} msg={data.get('message') or data.get('wording')}")
        return data.get("data", data)

    # ------------------------------------------------------------------ 便捷方法
    async def send_private_msg(self, user_id: int | str,
                               message: str | list[dict[str, Any]]) -> dict[str, Any]:
        """发私聊消息；message 为字符串时自动包成 text 段。"""
        msg = text_message(message) if isinstance(message, str) else message
        result = await self.call("send_private_msg", user_id=int(user_id), message=msg)
        return result if isinstance(result, dict) else {}

    async def send_group_msg(self, group_id: int | str,
                             message: str | list[dict[str, Any]]) -> dict[str, Any]:
        """发群消息；message 为字符串时自动包成 text 段。"""
        msg = text_message(message) if isinstance(message, str) else message
        result = await self.call("send_group_msg", group_id=int(group_id), message=msg)
        return result if isinstance(result, dict) else {}

    async def delete_msg(self, message_id: int | str) -> None:
        await self.call("delete_msg", message_id=int(message_id))

    async def get_login_info(self) -> dict[str, Any]:
        """取机器人自身账号信息（user_id / nickname）——用于「绑定」时校验。"""
        result = await self.call("get_login_info")
        return result if isinstance(result, dict) else {}

    async def get_status(self) -> dict[str, Any]:
        result = await self.call("get_status")
        return result if isinstance(result, dict) else {}

    async def get_version_info(self) -> dict[str, Any]:
        result = await self.call("get_version_info")
        return result if isinstance(result, dict) else {}

    async def get_group_list(self) -> list[dict[str, Any]]:
        """取加入的群列表（后台「可发布目标」页拉候选）。"""
        result = await self.call("get_group_list")
        return result if isinstance(result, list) else []

    async def get_group_info(self, group_id: int | str) -> dict[str, Any]:
        result = await self.call("get_group_info", group_id=int(group_id))
        return result if isinstance(result, dict) else {}

    async def get_group_member_list(self, group_id: int | str) -> list[dict[str, Any]]:
        result = await self.call("get_group_member_list", group_id=int(group_id))
        return result if isinstance(result, list) else []

    async def get_friend_list(self) -> list[dict[str, Any]]:
        """取好友列表（后台拉候选私聊目标）。"""
        result = await self.call("get_friend_list")
        return result if isinstance(result, list) else []