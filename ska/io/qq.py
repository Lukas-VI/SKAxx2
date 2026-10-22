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
from urllib.parse import urlsplit

from ..kernel.event import TOPIC_CHANNEL_STATUS, Event
from ..kernel.packet import CoreInput, CoreOutput
from ..media.images import build_message, send_dirs   # 有图输出时组装 OneBot 段数组
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
        self._proc: asyncio.subprocess.Process | None = None   # 自动拉起的 SnowLuma
        self._boot_task: asyncio.Task | None = None            # 自动拉起的等待任务
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
        # 自动拉起 SnowLuma（阶段三增强）：启用 QQ 时若探测不到适配器，就按命令启动它。
        # 只在本进程真正拉起时才会在 stop() 里关掉，绝不误杀用户手动启动的实例。
        self.auto_start = bool(cfg.get("auto_start", False))
        self.snowluma_command = str(cfg.get("snowluma_command", "") or "").strip()
        self.snowluma_cwd = str(cfg.get("snowluma_cwd", "") or "").strip()
        # SnowLuma 的 WebUI 端口：用于识别「已启动但还没登录 QQ」的实例，
        # 避免 OneBot 端口未开时错误地再拉起一个实例（双实例会撞 WebUI 端口）。
        try:
            self.snowluma_webui_port = max(int(cfg.get("snowluma_webui_port", 0) or 0), 0)
        except (TypeError, ValueError):
            self.snowluma_webui_port = 0
        try:
            self.startup_timeout = max(float(cfg.get("startup_timeout", 60)), 5.0)
        except (TypeError, ValueError):
            self.startup_timeout = 60.0
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
        # 自动启动被关掉（或渠道停用）→ 关掉**我们自己拉起**的 SnowLuma
        if not (self.enabled and self.auto_start):
            await self._stop_snowluma()
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
        self._stopped = False
        self.client = OneBotClient(self.api_base, self.access_token)
        # 自动拉起放后台：等待就绪最长 startup_timeout，不能阻塞整个应用启动
        # （WebUI 在渠道之后才起来，同步等 60s 会让管理后台迟迟打不开）。
        if self.auto_start:
            self._boot_task = asyncio.create_task(self._ensure_snowluma())
        self._task = asyncio.create_task(self._probe_loop())

    # ------------------------------------------------------------------ SnowLuma 自动启动
    async def _probe_once(self) -> bool:
        """单次连通性探测（不改变状态、不广播），供自动启动判断「是否已在运行」。"""
        if self.client is None:
            return False
        try:
            await self.client.get_login_info()
            return True
        except Exception:  # noqa: BLE001 —— 任何异常都等价于「不可用」
            return False

    async def _port_open(self, port: int) -> bool:
        """探测本机某个 TCP 端口是否已监听（用于识别「已启动但未登录」的 SnowLuma）。

        这里用纯 TCP 连接而不是 HTTP 请求：SnowLuma WebUI 可能需要登录/同意 EULA，
        任何 401/403 都仍说明「服务已起来」，而连接失败才是真的没起来。
        """
        host = urlsplit(self.api_base).hostname or "127.0.0.1"
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=1.0)
        except (OSError, asyncio.TimeoutError):
            return False
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001 —— 关闭失败不影响「端口是开的」这个结论
            pass
        return True

    async def _ensure_snowluma(self) -> bool:
        """确保 SnowLuma 可用：已在运行就直接用；否则按配置拉起并等它就绪。

        三个关键取舍：
            ① 先探测接口再启动——用户可能已经手动开了 SnowLuma，重复拉起会抢端口；
            ② 再探测 WebUI 端口——识别「已启动但未登录 QQ」的实例，同样不重复拉起；
            ③ 只等待到 startup_timeout，超时立即放弃并广播离线，
               不让渠道启动流程被一个起不来的外部进程永久拖住。
        """
        if await self._probe_once():
            logger.info("检测到 SnowLuma 已在运行（%s），跳过自动启动", self.api_base)
            return True
        # OneBot 端口未开，但 WebUI 已在监听 → SnowLuma 已启动、只是还没登录 QQ。
        # 此时绝不能再拉起第二个实例（会撞 WebUI 端口），直接提示去登录即可。
        if (self.snowluma_webui_port > 0
                and await self._port_open(self.snowluma_webui_port)):
            logger.info("检测到 SnowLuma WebUI（:%d）已在运行，跳过自动启动",
                        self.snowluma_webui_port)
            await self._status(
                "channel_offline",
                "SnowLuma 已在运行但尚未登录 QQ，请打开 WebUI 完成登录")
            return False
        if not self.snowluma_command:
            await self._status(
                "channel_offline",
                "QQ 已启用但探测不到 SnowLuma，且未配置 qq.snowluma_command（无法自动启动）")
            return False
        if not await self._spawn_snowluma():
            return False

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.startup_timeout
        while not self._stopped and loop.time() < deadline:
            await asyncio.sleep(1.0)
            if await self._probe_once():
                await self._status("channel_online", "SnowLuma 已自动启动并就绪")
                return True
            if self._proc is not None and self._proc.returncode is not None:
                await self._status(
                    "channel_offline",
                    f"SnowLuma 进程已退出（code={self._proc.returncode}），请检查启动命令")
                return False
        await self._status("channel_offline",
                           f"等待 SnowLuma 就绪超时（{self.startup_timeout:.0f}s）")
        return False

    async def _spawn_snowluma(self) -> bool:
        """按配置命令拉起 SnowLuma 子进程（shell 形式，便于带参数/路径空格）。"""
        try:
            self._proc = await asyncio.create_subprocess_shell(
                self.snowluma_command,
                cwd=self.snowluma_cwd or None,
                stdout=asyncio.subprocess.DEVNULL,   # 适配器日志与本进程日志解耦
                stderr=asyncio.subprocess.DEVNULL)
        except (OSError, ValueError) as e:
            await self._status("channel_offline", f"SnowLuma 启动失败：{e}")
            return False
        logger.info("已自动启动 SnowLuma：%s（pid=%s）",
                    self.snowluma_command, getattr(self._proc, "pid", "?"))
        return True

    async def _stop_snowluma(self) -> None:
        """关闭**本进程拉起**的 SnowLuma（外部实例的 _proc 为 None，不会被动到）。"""
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.terminate()                     # 先温和关闭（Windows 等价于 kill）
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
            logger.info("已关闭自动启动的 SnowLuma（pid=%s）", getattr(proc, "pid", "?"))
        except ProcessLookupError:
            pass                                # 进程已自己退出
        except Exception:  # noqa: BLE001 —— 退出流程不能因清理失败而中断
            logger.debug("关闭 SnowLuma 进程失败", exc_info=True)

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
        # 只关闭本进程自动拉起的 SnowLuma；用户手动启动的实例保持不动
        await self._stop_snowluma()
        if self.client is not None:
            try:
                await self.client.aclose()
            except Exception:  # noqa: BLE001
                logger.debug("关闭 OneBot 客户端失败", exc_info=True)

    async def _cancel_task(self) -> None:
        for task in (self._task, self._boot_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        self._task = None
        self._boot_task = None

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
            # 图片/表情包段：交给 Mind 做视觉解析（普通图片）或入库（表情包）
            "images": self._extract_images(ev),
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
                    # sub_type=1 或摘要含「表情」→ 用户发的是表情包，文本占位要区分开，
                    # 否则模型会把表情包当成一张普通照片来回应。
                    data = seg.get("data") or {}
                    is_sticker = (str(data.get("sub_type") or "") == "1"
                                  or "表情" in str(data.get("summary") or ""))
                    parts.append("[表情包]" if is_sticker else "[图片]")
                elif seg.get("type") in ("face", "mface"):
                    parts.append("[表情]")
            text = "".join(parts).strip()
            if text:
                return text
        if isinstance(message, str) and message.strip():
            return message
        return str(ev.get("raw_message", "") or "")

    @staticmethod
    def _extract_images(ev: dict[str, Any]) -> list[dict[str, Any]]:
        """抽取消息里的图片/表情段（供视觉解析与表情包入库）。

        OneBot array 格式常见字段：
            image: {"file": "...", "url": "http://...", "sub_type": 0|1, "summary": "..."}
            mface/face: {"id": "...", "summary": "..."}
        url 可能缺失（部分实现只给 file），只有 http(s) 的 file 才能当远程地址用。
        """
        message = ev.get("message")
        if not isinstance(message, list):
            return []
        out: list[dict[str, Any]] = []
        for seg in message:
            if not isinstance(seg, dict):
                continue
            typ = str(seg.get("type") or "")
            data = seg.get("data") or {}
            if typ == "image":
                url = str(data.get("url") or "").strip()
                file = str(data.get("file") or "").strip()
                if not url and file.lower().startswith(("http://", "https://")):
                    url = file
                sub = str(data.get("sub_type") or "")
                summary = str(data.get("summary") or "").strip()
                kind = "sticker" if (sub == "1" or "表情" in summary) else "image"
                out.append({"kind": kind, "url": url, "file": file,
                            "sub_type": sub, "summary": summary})
            elif typ in ("face", "mface"):
                out.append({
                    "kind": typ,
                    "id": str(data.get("id") or data.get("face_id") or ""),
                    "summary": str(data.get("summary") or "").strip(),
                    "url": str(data.get("url") or "").strip(),
                })
        return out

    # ------------------------------------------------------------------ 发送
    async def send(self, out: CoreOutput) -> bool:
        """发送一条消息；返回是否成功（供主循环做熔断统计）。

        有图时组装成 OneBot 段数组（文本 + image 段）；没有图仍走原来的纯字符串路径，
        保证旧行为（以及旧测试桩）完全不变。
        """
        if self.client is None:
            self.client = OneBotClient(self.api_base, self.access_token)
        message: str | list[dict[str, Any]] = out.content
        if out.images:
            media = self._media_cfg()
            message = build_message(
                out.content, out.images,
                max_bytes=int(media.get("max_image_bytes", 5 * 1024 * 1024)),
                allow_dirs=send_dirs(media),   # 白名单 + 表情包目录（见 images.send_dirs）
                max_images=int(media.get("max_images_per_message", 4)))
            if not message:                 # 图全被拒 + 没有文本 → 退回纯文本（可能为空）
                message = out.content
        try:
            if out.target == "group_msg":
                await self.client.send_group_msg(out.group_id or self.default_group, message)
            else:
                await self.client.send_private_msg(out.user or self.default_user, message)
            return True
        except (OneBotError, ValueError) as e:      # ValueError：目标 ID 非数字
            logger.warning("QQ 发送失败：%s", e)
            return False

    # ------------------------------------------------------------------ 辅助
    def _media_cfg(self) -> dict[str, Any]:
        """读取全局 media 小节（图片大小上限、允许的本地目录等）。

        与 `section()` 的区别：`section()` 读的是**本渠道**的小节，而图片相关参数
        是跨渠道共享的，统一放在顶层 `media:`。同样优先 RuntimeSettings，
        这样后台改完立即生效（无需重启）。
        """
        if self.deps.settings is not None:
            return self.deps.settings.section("media")
        return self.deps.cfg.section("media")

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