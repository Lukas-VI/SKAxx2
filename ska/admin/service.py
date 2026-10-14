"""AdminService：后台业务逻辑聚合层（阶段二新增，见方案 §5.1）。

它把分散的部件收拢成「后台能直接调用的一组动作」：
    * AdminStore   —— 后台数据（账号/设置/用量/用户/白名单/工具权限）
    * MemoryStore  —— 对话与记忆（记忆管理页）
    * RuntimeSettings —— 运行时设置（热改配置）
    * ProviderManager —— LLM 提供商（API 管理页）
    * app（装配结果）—— 渠道 / Mind / Beat / 插件 / 工具（运行状态与启停）

分工：`ska/web/api.py` 只做「路由 + 鉴权 + 入参校验」，
真正的读写与副作用都在这里，便于单测（可直接构造 service 打桩）。
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import pkgutil
from typing import Any

from ..io.onebot import OneBotClient, OneBotError
from .policy import LEVELS

logger = logging.getLogger(__name__)


class AdminService:
    """后台动作集合。"""

    def __init__(self, *, app: dict[str, Any], store: Any, memory: Any,
                 settings: Any, providers: Any, policy: Any) -> None:
        self.app = app                  # main.build() 的产物
        self.store = store              # AdminStore
        self.memory = memory            # MemoryStore（可为 None）
        self.settings = settings        # RuntimeSettings
        self.providers = providers      # ProviderManager
        self.policy = policy            # Policy

    # ================================================================== 运行状态
    async def status(self) -> dict[str, Any]:
        """后台「运行状态」：提供商、心跳、队列、渠道在线情况、各库行数。"""
        bus = self.app.get("bus")
        beat = self.app.get("beat")
        channels = self.app.get("channels", {})
        return {
            "provider": self.providers.active_info(),
            "beat": {
                "count": getattr(beat, "beat_count", 0),
                "bpm": beat.current_bpm() if beat is not None else self.settings.get("bpm", 6),
                "queue": bus.qsize() if bus is not None else 0,
            },
            "channels": {
                name: {"online": bool(getattr(ch, "connected", True))}
                for name, ch in channels.items()
            },
            "memory": self.memory.stats() if self.memory is not None else {},
            "admin": await self.store.stats(),
            "provider_count": len(self.providers.list()),
        }

    # ================================================================== 设置
    async def settings_snapshot(self) -> dict[str, Any]:
        return self.settings.snapshot()

    async def update_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
        """合并式保存设置：写库 → 更新内存快照 → 尽可能热生效。"""
        over = self.settings.overrides()
        for section in ("throttle", "memory", "agent", "qq", "web", "roles"):
            part = payload.get(section)
            if isinstance(part, dict):
                over.setdefault(section, {}).update(part)
        app_part = payload.get("app")
        if isinstance(app_part, dict):
            over.setdefault("app", {}).update(app_part)
        await self.store.save_settings(over)
        self.settings.set_overrides(over)
        await self._apply_live(payload)
        return self.settings.snapshot()

    async def _apply_live(self, payload: dict[str, Any]) -> None:
        """把能热生效的部分立刻应用（其余项在下一次读取时自然生效）。"""
        # 人设档变化 → 让 Mind 重建 system prompt
        app_part = payload.get("app") or {}
        if "persona_profile" in app_part:
            mind = self.app.get("mind")
            refresh = getattr(mind, "refresh_persona", None)
            if refresh:
                refresh()
        # 最大步数变化 → 直接改执行者的上限
        if "agent" in payload and isinstance(payload["agent"], dict):
            if "max_steps" in payload["agent"]:
                agent = self.app.get("agent")
                if agent is not None:
                    try:
                        agent.max_steps = max(int(payload["agent"]["max_steps"]), 1)
                    except (TypeError, ValueError):
                        pass
        # QQ 参数变化 → 重连渠道
        if "qq" in payload:
            await self.reload_qq()

    # ================================================================== 插件
    def _available_plugins(self) -> list[str]:
        """枚举 ska/plugins 下的可用插件模块（排除 __init__）。"""
        pkg = importlib.import_module("ska.plugins")
        return sorted(info.name.rsplit(".", 1)[-1]
                      for info in pkgutil.iter_modules(pkg.__path__)
                      if not info.name.startswith("_"))

    async def list_plugins(self) -> list[dict[str, Any]]:
        enabled = set(self.settings.names("plugins"))
        loaded = set(self.app.get("plugin_map", {}))
        return [{"name": name, "enabled": name in enabled, "loaded": name in loaded}
                for name in self._available_plugins()]

    async def toggle_plugin(self, name: str, enabled: bool) -> dict[str, Any]:
        """启停插件：写设置 + 立即装载/卸载（scheduler 等可在运行期安全启停）。"""
        if name not in self._available_plugins():
            raise ValueError(f"未知插件：{name}")
        names = self.settings.names("plugins")
        if enabled and name not in names:
            names.append(name)
        if not enabled and name in names:
            names.remove(name)
        over = self.settings.overrides()
        over.setdefault("app", {})["plugins"] = names
        await self.store.save_settings(over)
        self.settings.set_overrides(over)

        if enabled:
            await self._install_plugin(name)
        else:
            await self._uninstall_plugin(name)
        return {"name": name, "enabled": enabled,
                "loaded": name in self.app.get("plugin_map", {})}

    async def _install_plugin(self, name: str) -> None:
        plugin_map = self.app.setdefault("plugin_map", {})
        if name in plugin_map:
            return
        module = importlib.import_module(f"ska.plugins.{name}")
        plugin_map[name] = module.install(self.app["bus"], self.app["cfg"])
        logger.info("插件已装载：%s", name)

    async def _uninstall_plugin(self, name: str) -> None:
        instance = self.app.get("plugin_map", {}).pop(name, None)
        if instance is None:
            return
        stopper = getattr(instance, "stop", None)
        if stopper:
            try:
                result = stopper()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001
                logger.exception("插件卸载异常：%s", name)
        logger.info("插件已卸载：%s", name)

    # ================================================================== 工具 / 智能体
    async def list_tools(self) -> list[dict[str, Any]]:
        """列出已装配工具及其开关/最小等级（未单独配置的视为默认放行）。"""
        built = self.app.get("tools", {})
        perms = {p["tool"]: p for p in await self.store.list_tool_permissions()}
        items = []
        for name, tool in built.items():
            perm = perms.get(name, {})
            items.append({
                "name": name,
                "description": getattr(tool, "description", ""),
                "enabled": bool(perm.get("enabled", 1)),
                "min_level": str(perm.get("min_level", "member")),
                "configured": name in perms,
            })
        return items

    async def update_tool(self, name: str, *, enabled: bool, min_level: str,
                          note: str = "") -> dict[str, Any]:
        if min_level not in LEVELS:
            raise ValueError(f"非法等级：{min_level}（可选：{', '.join(LEVELS)}）")
        await self.store.upsert_tool_permission(name, enabled, min_level, note)
        return {"name": name, "enabled": enabled, "min_level": min_level}

    async def agent_get(self) -> dict[str, Any]:
        return self.settings.agent()

    async def agent_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = await self.update_settings({"agent": payload})
        return result["agent"]

    # ================================================================== 权限 / 角色
    async def roles_get(self) -> dict[str, Any]:
        return {"levels": list(LEVELS), **self.settings.roles()}

    async def roles_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = await self.update_settings({"roles": payload})
        return {"levels": list(LEVELS), **result["roles"]}

    # ================================================================== 聊天用户
    async def list_users(self) -> list[dict[str, Any]]:
        return await self.store.list_chat_users()

    async def update_user(self, channel: str, user_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        level = payload.get("level")
        if level is not None and level not in LEVELS:
            raise ValueError(f"非法等级：{level}")
        blacklisted = payload.get("blacklisted")
        await self.store.update_chat_user(
            channel, user_id,
            level=level,
            blacklisted=None if blacklisted is None else bool(blacklisted),
            note=payload.get("note"),
        )
        row = await self.store.get_chat_user(channel, user_id)
        return row or {}

    # ================================================================== 记忆
    async def memory_long(self, limit: int = 100, offset: int = 0,
                          user: str = "") -> list[dict[str, Any]]:
        return await self.memory.list_long(limit=limit, offset=offset, user=user)

    async def memory_add(self, content: str, importance: float = 0.5,
                         user: str = "") -> dict[str, Any]:
        mem_id = await self.memory.add_long(content, source="manual",
                                            importance=float(importance), user=user)
        return {"id": mem_id}

    async def memory_update(self, mem_id: int, payload: dict[str, Any]) -> bool:
        return await self.memory.update_long(
            mem_id, content=payload.get("content"),
            importance=payload.get("importance"))

    async def memory_delete(self, mem_id: int) -> bool:
        return await self.memory.delete_long(mem_id)

    async def memory_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        return await self.memory.list_sessions(limit=limit)

    async def memory_session_delete(self, session_id: int) -> bool:
        return await self.memory.delete_session(session_id)

    async def memory_profiles(self) -> list[dict[str, Any]]:
        return await self.memory.list_profiles()

    async def memory_messages(self, limit: int = 100, user: str = "") -> list[dict[str, Any]]:
        return await self.memory.list_messages(limit=limit, user=user)

    async def memory_message_delete(self, msg_id: int) -> bool:
        return await self.memory.delete_message(msg_id)

    # ================================================================== 可发布目标
    async def list_targets(self) -> list[dict[str, Any]]:
        return await self.store.list_targets()

    async def add_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        target_type = str(payload.get("target_type") or "group")
        if target_type not in ("group", "user"):
            raise ValueError("target_type 只能是 group 或 user")
        target_id = str(payload.get("target_id") or "").strip()
        if not target_id:
            raise ValueError("target_id 不能为空")
        row_id = await self.store.add_target(
            str(payload.get("channel") or "qq"), target_type, target_id,
            str(payload.get("name") or ""), bool(payload.get("enabled", True)),
            str(payload.get("note") or ""))
        return {"id": row_id}

    async def update_target(self, row_id: int, payload: dict[str, Any]) -> bool:
        return await self.store.update_target(
            row_id, enabled=payload.get("enabled"),
            name=payload.get("name"), note=payload.get("note"))

    async def delete_target(self, row_id: int) -> bool:
        return await self.store.delete_target(row_id)

    # ================================================================== 用量
    async def usage(self, days: int = 7, group_by: str = "provider") -> dict[str, Any]:
        summary = await self.store.usage_summary(days=days, group_by=group_by)
        summary["recent"] = await self.store.usage_recent(limit=30)
        return summary

    async def record_usage(self, provider: str, model: str,
                           usage: dict[str, Any], task: str) -> None:
        """LLMGateway.on_usage 的回调实现（网关照例调用，异常不冒泡）。"""
        if not isinstance(usage, dict):
            return
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or (prompt + completion))
        await self.store.record_usage(provider, model, task or "chat",
                                      prompt, completion, total)

    # ================================================================== QQ 适配器
    def _qq_channel(self):
        return self.app.get("channels", {}).get("qq")

    async def reload_qq(self) -> dict[str, Any]:
        """按最新配置重连 QQ 渠道（后台改完参数即生效）。"""
        channel = self._qq_channel()
        if channel is None:
            return {"ok": False, "error": "当前未装配 qq 渠道（请把 qq 加入 channels）"}
        await channel.reload()
        cfg = self.settings.qq()
        return {"ok": True, "enabled": bool(cfg.get("enabled")), "connected": channel.connected}

    async def qq_config(self) -> dict[str, Any]:
        cfg = self.settings.qq()
        channel = self._qq_channel()
        return {**cfg, "connected": bool(channel.connected) if channel else False}

    async def qq_update(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = await self.update_settings({"qq": payload})
        return result["qq"]

    async def qq_login_info(self) -> dict[str, Any]:
        """读取机器人账号信息（顺带做一次连通性检查）。"""
        cfg = self.settings.qq()
        return await self._probe(cfg)

    async def qq_test(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """用（可覆盖的）参数做连通性测试——不改动已保存配置。"""
        cfg = {**self.settings.qq(), **(payload or {})}
        return await self._probe(cfg)

    async def _probe(self, cfg: dict[str, Any]) -> dict[str, Any]:
        client = OneBotClient(str(cfg.get("api_base") or ""),
                              str(cfg.get("access_token") or ""), timeout=10)
        try:
            login = await client.get_login_info()
            status = await client.get_status()
            try:
                version = await client.get_version_info()
            except OneBotError:
                version = {}
            return {"ok": True, "login": login, "status": status, "version": version}
        except OneBotError as e:
            return {"ok": False, "error": str(e)}
        finally:
            await client.aclose()

    async def qq_candidates(self) -> dict[str, Any]:
        """从 SnowLuma 拉取「可作为主动发言目标」的群与好友（结合适配器 API 拓展）。"""
        cfg = self.settings.qq()
        client = OneBotClient(str(cfg.get("api_base") or ""),
                              str(cfg.get("access_token") or ""), timeout=15)
        try:
            groups = await client.get_group_list()
            friends = await client.get_friend_list()
            return {
                "ok": True,
                "groups": [{"id": str(g.get("group_id", "")),
                            "name": str(g.get("group_name", ""))} for g in groups],
                "friends": [{"id": str(f.get("user_id", "")),
                             "name": str(f.get("nickname", ""))} for f in friends],
            }
        except OneBotError as e:
            return {"ok": False, "error": str(e), "groups": [], "friends": []}
        finally:
            await client.aclose()