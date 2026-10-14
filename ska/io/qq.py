"""QQ 渠道：默认对接 SnowLuma 适配器（阶段二改造）。

阶段一用 SSE（`GET /_events`）自建接收循环（NapCat 风格）；
但 SnowLuma **不提供 SSE**，只有 OneBot v11 的 HTTP/WS 调用。因此本渠道改为
「事件注入式」（决策 D3，见方案 §5.4）：

    收事件：SnowLuma 的 httpClients 把事件 POST 到本机 `/onebot/events`，
            由 Web 线程 → asyncio 桥交回事件循环，再调用 `QQChannel.on_event(raw)`；
    发信  ：走 SnowLuma 的 httpServers（HTTP POST `send_group_msg` / `send_private_msg`）。

这样做的额外好处：
    * 渠道不再自己维护长连接与重连逻辑，`start()` 只需「探测连通性 + 广播状态」；
    * 后台「绑定 QQ」页可以用同一套配置做连通性测试与账号信息展示。

保留阶段一的两处修复：
    1. 每条消息立即入队（不放在单槽位里），不会因下游处理慢而丢消息；
    2. 断连用指数退避重试，并通过 channel_status 事件告知 Mind。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..kernel.event import TOPIC_CHANNEL_STATUS, Event
from ..kernel.packet import CoreInput, CoreOutput
from .base import Channel, channel
from .onebot import OneBotClient, OneBotError

logger = logging.getLogger(__name__)


@channel("qq")
class QQChannel(Channel):
    """OneBot v11 渠道（默认 SnowLuma）。"""

    def __init__(self, deps) -> None:
        super().__init__(deps)
        self.client: OneBotClient | None = None
        self._task: asyncio.Task | None = None
        self._stopped = False
        self._connected = False
        self._backoff = 1.0
        self._load_cfg()                     # 从 RuntimeSettings 载入配置并建客户端

    # ------------------------------------------------------------------ 配置
    def _load_cfg(self) -> None:
        """从配置小节载入连接参数（支持后台热改后调用 reload 重新载入）。"""
        cfg = self.section()
        self.enabled = bool(cfg.get("enabled", False))
        self.transport = str(cfg.get("transport", "http"))          # http（默认）| ws（预留）
        self.api_base = str(cfg.get("api_base", "http://127.0.0.1:3000")).rstrip("/")
        self.access_token = str(cfg.get("access_token", "") or "")
        self.report_token = str(cfg.get("report_token", "") or "")
        self.self_id = str(cfg.get("self_id", "") or "")
        self.default_group = str(cfg.get("default_group", "") or "")
        self.default_user = str(cfg.get("default_user", "") or "")
        self.backoff_min = max(float(cfg.get("reconnect_min", 1)), 0.1)
        self.backoff_max = max(float(cfg.get("reconnect_max", 60)), self.backoff_min)
        self._backoff = self.backoff_min

    async def reload(self) -> None:
        """后台改完 QQ 配置后调用：停掉旧探测任务，按新配置重连（热生效）。"""
        await self._cancel_task()
        old = self.client
        self.client = None
        if old is not None:
            try:
                await old.aclose()           # 关闭旧连接池（token 变更必须重建）
            except Exception:  # noqa: BLE001
                logger.debug("关闭旧 OneBot 客户端失败", exc_info=True)
        self._connected = False
        self._load_cfg()
        if self.enabled:
            await self.start()
        else:
            await self._status("channel_offline", "QQ 渠道已停用")

    # ------------------------------------------------------------------ 启动/停止
    async def start(self) -> None:
        """启动连通性探测（不建立接收循环——事件由 /onebot/events 注入）。

        ws 传输本阶段仅保留接口：SnowLuma 的 wsServers 需要自行实现握手与帧解析，
        按方案「阶段二默认不启用，留接口」处理。
        """
        if self.transport == "ws":
            logger.warning("qq transport=ws 尚未实现（本阶段默认 http），渠道保持离线")
            await self._status("channel_offline", "QQ ws 传输未实现，请改用 http")
            return
        if not self.enabled:
            logger.info("QQ 渠道未启用（可在后台「绑定 QQ」页开启）")
            return
        self.client = OneBotClient(self.api_base, self.access_token)
        self._task = asyncio.create_task(self._probe_loop())

    async def _probe_loop(self) -> None:
        """周期性探测连通性：成功→在线；失败→断线并指数退避。"""
        while not self._stopped:
            try:
                info = await self.client.get_login_info()       # 最轻量的连通性探测
                just = not self._connected
                self._connected = True
                self._backoff = self.backoff_min
                self.self_id = str(info.get("user_id") or self.self_id)   # 自动补全机器人账号
                if just:
                    nick = info.get("nickname") or ""
                    await self._status("channel_online", f"QQ 渠道已连接（{nick} {self.self_id}）")
            except OneBotError as e:
                if self._connected:                             # 从在线变离线才广播
                    await self._status("channel_offline", f"QQ 渠道断开：{e}")
                self._connected = False
                if self._stopped:
                    break
                await asyncio.sleep(self._backoff)
                self._backoff = min(self._backoff * 2, self.backoff_max)
                continue
            # 在线时每 30 秒复查一次（SnowLuma 无事件流，靠轮询感知掉线）
            await asyncio.sleep(30)
        self._connected = False

    async def stop(self) -> None:
        self._stopped = True
        await self._cancel_task()
        if self.client is not None:
            try:
                await self.client.aclose()
            except Exception:  # noqa: BLE001
                logger.debug("关闭 OneBot 客户端失败", exc_info=True)

    async def _cancel_task(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        self._task = None

    # ------------------------------------------------------------------ 接收（事件注入）
    async def on_event(self, raw: dict[str, Any]) -> bool:
        """处理一条由 SnowLuma 上报的原始事件；返回是否被接受。

        只有 message 类型、且不是机器人自己发的消息，才会转成 CoreInput 入队。
        立即 create_task 入队（不 await 下游），确保高并发下不丢消息。
        """
        if not isinstance(raw, dict):
            return False
        if raw.get("post_type") != "message":
            return False                       # 忽略 meta_event / notice / request 等
        if self.self_id and str(raw.get("user_id", "")) == self.self_id:
            return False                       # 忽略机器人自己的消息，避免自循环
        packet = self._to_input(raw)
        asyncio.create_task(self.publish_input(packet))
        return True

    def _to_input(self, ev: dict[str, Any]) -> CoreInput:
        """把 OneBot v11 消息事件转成 CoreInput。

        文本抽取顺序：消息段数组（只取 text，@某人转成可读文本）→ 字符串 message
        → raw_message（含 CQ 码，最后兜底）。
        """
        sender = ev.get("sender", {}) or {}
        msg_type = str(ev.get("message_type", "private"))
        fields: dict[str, Any] = {
            "source": "qq_message",
            "content": self._extract_text(ev),
            "user": str(ev.get("user_id", "")),
            "user_name": str(sender.get("card") or sender.get("nickname") or ""),
            "channel": self.name,
            "message_type": "group" if msg_type == "group" else "private",
            "group_id": str(ev.get("group_id", "")) if msg_type == "group" else "",
            "raw": ev,
        }
        # 事件 time 缺失/为 0 时不要传 ts（否则 None 会触发 pydantic 校验错误，
        # 交给 CoreInput 的默认值 time.time() 接管）
        raw_ts = float(ev.get("time", 0) or 0)
        if raw_ts > 0:
            fields["ts"] = raw_ts
        return CoreInput(**fields)

    @staticmethod
    def _extract_text(ev: dict[str, Any]) -> str:
        """从事件里抽取可读文本（兼容 array / string 两种 messageFormat）。"""
        message = ev.get("message")
        if isinstance(message, list):
            parts: list[str] = []
            for seg in message:
                if not isinstance(seg, dict):
                    continue
                if seg.get("type") == "text":
                    parts.append(str((seg.get("data") or {}).get("text", "")))
                elif seg.get("type") == "at":
                    qq = (seg.get("data") or {}).get("qq")
                    parts.append(f"@{qq}" if qq else "@某人")
                elif seg.get("type") == "image":
                    parts.append("[图片]")
                elif seg.get("type") == "face":
                    parts.append("[表情]")
            text = "".join(parts).strip()
            if text:
                return text
        if isinstance(message, str) and message.strip():
            return message
        return str(ev.get("raw_message", "") or "")

    # ------------------------------------------------------------------ 发送
    async def send(self, out: CoreOutput) -> bool:
        """发送一条消息；返回是否成功（供主循环做熔断统计）。"""
        if self.client is None:
            self.client = OneBotClient(self.api_base, self.access_token)
        try:
            if out.target == "group_msg":
                await self.client.send_group_msg(out.group_id or self.default_group, out.content)
            else:
                await self.client.send_private_msg(out.user or self.default_user, out.content)
            return True
        except (OneBotError, ValueError) as e:      # ValueError：目标 ID 非数字
            logger.warning("QQ 发送失败：%s", e)
            return False

    # ------------------------------------------------------------------ 辅助
    async def _status(self, event: str, text: str) -> None:
        """广播渠道状态：作为输入类事件喂给 Mind（它可能想打招呼或说明情况）。"""
        logger.info("[qq] %s：%s", event, text)
        await self.bus.publish(Event(
            topic=TOPIC_CHANNEL_STATUS,
            payload=CoreInput(source="system_event", event=event, content=text,
                              channel=self.name),
        ))

    @property
    def connected(self) -> bool:
        """渠道在线状态（供后台「运行状态」页展示）。"""
        return self._connected