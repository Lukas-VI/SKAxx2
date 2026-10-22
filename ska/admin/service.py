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

import base64
import binascii
import logging
from pathlib import Path
from typing import Any

from ..io.onebot import OneBotClient, OneBotError
from ..kernel.config import load_yaml, write_persona_default
from ..llm.gateway import LLMError
from ..llm.models import ChatMessage
from ..llm.structured import StructuredError, structured_chat
from .policy import LEVELS

logger = logging.getLogger(__name__)

# 结构化事实抽取提示词：把一段领域知识拆成原子三元组（同槽位可覆盖更新）。
FACT_EXTRACT_SYSTEM = """你是知识库结构化助手。把用户提供的领域知识拆成尽量细的原子事实：
- subject：主体（实体名，尽量用名词）；
- predicate：关系或属性（短词，如「属于」「位于」「成分」「适合」）；
- object：客体或值；
- fact：一句完整的中文陈述；
- importance：0~1 的重要性。
要求：一条事实只表达一件事；同一主体的同类关系尽量用同一个 predicate。
只输出 JSON，不要解释：
{"facts":[{"subject":"","predicate":"","object":"","fact":"","importance":0.6}]}"""


def _decode_upload(data: str) -> bytes:
    """解码前端上传的图片：支持裸 base64 与 data:image/...;base64,xxx 两种。"""
    raw = str(data or "").strip()
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]                # 去掉 data URL 头
    return base64.b64decode(raw, validate=False)


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

    # ================================================================== 首次初始化
    def ensure_local_config(self) -> list[str]:
        """补齐「本地私有配置」文件，返回本次**新建**的文件名列表。

        这两个文件不入库（见 .gitignore），克隆仓库后并不存在；后台「首次初始化」
        调用它，做到「克隆 → 启动 → 初始化」即可直接使用：
            * config/persona.yaml   —— 内置通用人设模板（用户随后可自行改写）
            * config/providers.yaml —— 提供商清单与密钥（ProviderManager 负责落盘）

        已存在的文件一律不动，避免覆盖用户自己的内容。
        """
        created: list[str] = []

        # 1) 人设模板
        try:
            if write_persona_default():
                created.append("config/persona.yaml")
        except Exception:  # noqa: BLE001 —— 生成失败不应阻断初始化流程
            logger.exception("生成 config/persona.yaml 失败")

        # 2) 提供商清单。文件由 ProviderManager 在首次启动时就已落盘，
        #    这里再存一次，保证初始化期间在向导里填的密钥被写入文件。
        prov_path = getattr(self.providers, "path", None)
        existed = bool(prov_path is not None and Path(prov_path).exists())
        try:
            self.providers.save()
            if not existed:
                created.append("config/providers.yaml")
        except Exception:  # noqa: BLE001
            logger.exception("生成 config/providers.yaml 失败")

        # 3) 人设模板刚写出来时，让 Mind 立刻按新模板重建 system prompt，
        #    否则要等到下次重启才生效（此时 cfg.persona 还停留在空档）。
        if any(name.endswith("persona.yaml") for name in created):
            cfg = self.app.get("cfg")
            if cfg is not None:
                cfg.persona = load_yaml("config/persona.yaml", required=False)
            refresh = getattr(self.app.get("mind"), "refresh_persona", None)
            if refresh:
                try:
                    refresh()
                except Exception:  # noqa: BLE001
                    logger.exception("按新人设重建 system prompt 失败")

        return created

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
        for section in ("throttle", "memory", "agent", "qq", "web", "roles", "webchat"):
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
        # webchat 参数变化 → 立刻把默认身份 / 落库开关同步给 ChatHub（新建会话即生效）
        if "webchat" in payload:
            self._sync_chat_defaults()

    def _sync_chat_defaults(self) -> None:
        """把 webchat 小节的默认身份与 persist 同步到 ChatHub 与渠道（热改生效）。"""
        hub = self.app.get("chat_hub")
        if hub is None:
            return
        cfg = self.settings.webchat()
        hub.default_identity = str(cfg.get("identity") or cfg.get("user") or "owner")
        hub.default_persist = bool(cfg.get("persist", True))
        channel = self.app.get("channels", {}).get("webchat")
        if channel is not None:
            channel.hub = hub

    # ================================================================== 插件
    def _plugin_manager(self) -> Any:
        """取插件运行时管理器（main.build() 已建好；测试里缺就惰性补一个）。"""
        manager = self.app.get("plugin_manager")
        if manager is None:
            from ..plugins.manager import PluginManager
            manager = PluginManager(self.app, settings=self.settings, store=self.store)
            self.app["plugin_manager"] = manager
        return manager

    async def list_plugins(self) -> list[dict[str, Any]]:
        return self._plugin_manager().describe()

    async def toggle_plugin(self, name: str, enabled: bool) -> dict[str, Any]:
        """启停插件：写设置 + 立即装载/卸载（scheduler 等可在运行期安全启停）。"""
        return await self._plugin_manager().toggle(name, enabled)

    async def reload_plugin(self, name: str) -> dict[str, Any]:
        """热重载：重新读取磁盘源码并重新装载（改完插件文件后一键生效）。"""
        manager = self._plugin_manager()
        await manager.reload(name)
        return {"name": name, "loaded": name in manager.loaded(),
                "plugins": manager.describe()}

    async def delete_plugin(self, name: str) -> dict[str, Any]:
        """卸载并删除插件文件（内置插件受保护，会抛错）。"""
        manager = self._plugin_manager()
        await manager.delete(name)
        return {"name": name, "deleted": True, "plugins": manager.describe()}

    async def create_plugin(self, name: str, source: str,
                            *, enabled: bool = True) -> dict[str, Any]:
        """新建/覆写插件源码并热装载（WebUI「新建插件」用）。"""
        manager = self._plugin_manager()
        result = await manager.create(name, source, enabled=enabled)
        return {**result, "plugins": manager.describe()}

    async def plugin_source(self, name: str) -> dict[str, Any]:
        """读取插件源码（后台编辑框「载入」用；只读，不改状态）。"""
        return {"name": name, "source": self._plugin_manager().read_source(name)}

    async def plugin_settings(self, name: str) -> dict[str, Any]:
        """读取插件设置项：schema（前端据此生成表单）+ 当前生效值 + 可选值。

        「可选值」只对 type=targets 的字段生成：取自可发布目标表，外加 QQ 默认群
        （默认消息通道就是群聊，即使还没登记成可发布目标也应能选）。
        """
        manager = self._plugin_manager()
        schema = manager.settings_schema(name)
        if not schema:
            return {"name": name, "schema": {}, "values": {}, "options": {}}
        values = manager.default_settings(schema)
        if self.settings is not None:
            values.update(self.settings.section(f"plugin:{name}"))
        options: dict[str, Any] = {}
        if any(str(f.get("type")) == "targets" for f in schema.get("fields") or []):
            rows = await self.store.list_targets() if self.store is not None else []
            items = [{
                "value": f"{r.get('channel')}|{r.get('target_type')}|{r.get('target_id')}",
                "label": (f"{r.get('name') or r.get('target_id')}"
                          f"（{r.get('channel')}·{'群' if r.get('target_type') == 'group' else '好友'}）"),
            } for r in rows if r.get("enabled", 1)]
            seen = {item["value"] for item in items}
            group = str((self.settings.qq() if self.settings is not None else {})
                        .get("default_group") or "").strip()
            if group and f"qq|group|{group}" not in seen:
                items.insert(0, {"value": f"qq|group|{group}",
                                 "label": f"{group}（qq·群·默认）"})
            options["targets"] = items
        return {"name": name, "schema": schema, "values": values, "options": options}

    async def update_plugin_settings(self, name: str,
                                     values: dict[str, Any]) -> dict[str, Any]:
        """保存插件设置项（校验 → 写库 → 热重载插件，使其立即按新值工作）。"""
        manager = self._plugin_manager()
        clean = manager.validate_settings(name, values)
        if self.settings is None or self.store is None:
            raise ValueError("后台设置存储不可用，无法保存插件设置")
        over = self.settings.overrides()
        over[f"plugin:{name}"] = clean
        await self.store.save_settings(over)
        self.settings.set_overrides(over)
        if name in manager.loaded():          # 运行中就重载，让新设置立刻生效
            await manager.reload(name)
        return await self.plugin_settings(name)

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

    # 核心记忆（core_facts）：后台需要看到「Agent 始终记得的事」，
    # 否则会出现「它说记住了，但记忆管理里找不到」的对不上号问题。
    async def memory_core(self, limit: int = 100, user: str = "") -> list[dict[str, Any]]:
        return await self.memory.list_core_entries(user=user, limit=limit)

    async def memory_core_add(self, content: str, importance: float = 0.8,
                              user: str = "") -> dict[str, Any]:
        core_id = await self.memory.pin_core(content, user=user,
                                             importance=float(importance))
        return {"id": core_id}

    async def memory_core_delete(self, core_id: int) -> bool:
        return await self.memory.delete_core(core_id)

    # 结构化事实（entity_facts）：摘要反思抽出的三元组，后台按槽位展示。
    async def memory_facts(self, limit: int = 200, user: str = "") -> list[dict[str, Any]]:
        return await self.memory.list_facts(user=user, limit=limit)

    async def memory_fact_delete(self, fact_id: int) -> bool:
        return await self.memory.delete_fact(fact_id)

    # ---- 结构化事实导入（把某个领域的细分知识交给模型拆成原子事实）----
    async def memory_facts_import(self, items: list[dict[str, Any]], *,
                                  user: str = "") -> dict[str, int]:
        """直接导入一组已写好的三元组；同槽位按 upsert 语义覆盖。"""
        added = updated = skipped = 0
        for item in items[:200]:              # 一次最多 200 条，防止误导入海量数据
            if not isinstance(item, dict):
                skipped += 1
                continue
            subject = str(item.get("subject") or "").strip()
            predicate = str(item.get("predicate") or "").strip()
            obj = str(item.get("object") or "").strip()
            if not (subject and predicate and obj):
                skipped += 1
                continue
            try:
                importance = float(item.get("importance", 0.6))
            except (TypeError, ValueError):
                importance = 0.6
            state = await self.memory.upsert_fact(
                subject, predicate, obj, user=user, importance=importance,
                fact=str(item.get("fact") or ""))
            if state == "added":
                added += 1
            else:
                updated += 1
        return {"added": added, "updated": updated, "skipped": skipped}

    async def memory_facts_extract(self, text: str, *, user: str = "",
                                   domain: str = "") -> dict[str, Any]:
        """让模型把一段领域知识拆成结构化事实，再批量导入。

        这是「导入某个领域细分知识」的入口：用户只需贴一段资料（或说明领域），
        模型负责拆成 subject/predicate/object，后续检索/覆盖更新沿用既有机制。
        """
        body = str(text or "").strip()
        if not body:
            raise ValueError("知识原文不能为空")
        gateway = self.app.get("gateway")
        if gateway is None:
            raise ValueError("LLM 网关未就绪")
        prompt = f"领域：{domain.strip() or '通用'}\n知识原文：\n{body[:8000]}"
        try:
            data = await structured_chat(
                gateway,
                [ChatMessage(role="system", content=FACT_EXTRACT_SYSTEM),
                 ChatMessage(role="user", content=prompt)],
                task="fact_extract")
        except (LLMError, StructuredError) as e:
            raise ValueError(f"模型未能输出结构化事实：{e}") from e
        items = data.get("facts") if isinstance(data, dict) else None
        if not isinstance(items, list) or not items:
            raise ValueError("模型没有拆出任何事实（可换种说法或补全知识原文）")
        result = await self.memory_facts_import(items, user=user)
        result["facts"] = [i for i in items if isinstance(i, dict)][:50]
        return result

    # ================================================================== 表情包管理
    async def stickers(self, limit: int = 200) -> list[dict[str, Any]]:
        book = self.app.get("stickers")
        if book is None:
            return []
        return await book.list(limit=limit)

    async def sticker_update(self, sticker_id: int, payload: dict[str, Any]) -> bool:
        book = self.app.get("stickers")
        if book is None:
            raise ValueError("表情包库未启用")
        return await book.update(
            sticker_id,
            description=payload.get("description"),
            tags=payload.get("tags"),
            emotion=payload.get("emotion"))

    async def sticker_delete(self, sticker_id: int) -> bool:
        book = self.app.get("stickers")
        if book is None:
            raise ValueError("表情包库未启用")
        return await book.delete(sticker_id)

    async def sticker_image(self, sticker_id: int) -> tuple[bytes, str] | None:
        book = self.app.get("stickers")
        if book is None:
            return None
        return await book.image(sticker_id)

    async def stickers_import(self, items: list[dict[str, Any]], *,
                              user: str = "") -> dict[str, int]:
        """批量导入表情包：前端把文件夹里的每张图读成 base64 一起提交。

        单张失败只记 failed，不影响整批；重复图片（sha256 命中）记为 skipped。
        """
        book = self.app.get("stickers")
        if book is None:
            raise ValueError("表情包库未启用")
        added = skipped = failed = 0
        for item in items[:200]:              # 单次请求上限，前端会分批提交
            if not isinstance(item, dict):    # 前端传坏结构：计入失败，不影响整批
                failed += 1
                continue
            name = str(item.get("name") or "")
            raw = str(item.get("data") or "")
            try:
                blob = _decode_upload(raw)
            except (binascii.Error, ValueError):
                failed += 1
                continue
            if not blob:
                failed += 1
                continue
            try:
                result = await book.import_one(blob, name=name, user=user)
            except Exception:  # noqa: BLE001 —— 单张异常不中断整批
                logger.exception("导入表情包失败：%s", name)
                failed += 1
                continue
            status = result.get("status")
            if status == "added":
                added += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
        return {"added": added, "skipped": skipped, "failed": failed}

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

    # ================================================================== 在线聊天（阶段三 §2.4）
    def _chat_hub(self):
        """取 ChatHub（未装配时抛可读错误，由 API 层转成 4xx）。"""
        hub = self.app.get("chat_hub")
        if hub is None:
            raise ValueError("在线聊天不可用（ChatHub 未装配）")
        return hub

    async def chat_meta(self) -> dict[str, Any]:
        """聊天页状态条所需的元信息：身份 / 等级 / 默认落库 / 轮询间隔 / 模型窗口。"""
        cfg = self.settings.webchat()
        identity = str(cfg.get("identity") or cfg.get("user") or "owner")
        level = "member"
        if self.policy is not None:
            try:
                level = await self.policy.level_of("webchat", identity)
            except Exception:  # noqa: BLE001 —— 等级查询失败不该阻断页面
                logger.debug("读取聊天身份等级失败", exc_info=True)
        return {
            "enabled": bool(cfg.get("enabled", True)),
            "identity": identity,
            "level": level,
            "persist": bool(cfg.get("persist", True)),
            "poll_ms": int(cfg.get("poll_ms", 1000) or 1000),
            "context_window": self.providers.context_window(),
        }

    async def chat_list(self) -> list[dict[str, Any]]:
        return self._chat_hub().list()

    async def chat_create(self, payload: dict[str, Any]) -> dict[str, Any]:
        hub = self._chat_hub()
        session = hub.create(
            title=str(payload.get("title") or ""),
            persist=None if payload.get("persist") is None else bool(payload.get("persist")),
            identity=str(payload.get("identity") or ""),
            session_id=str(payload.get("id") or ""))
        return session.info()

    async def chat_update(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        hub = self._chat_hub()
        return hub.update(
            session_id,
            title=payload.get("title"),
            persist=None if payload.get("persist") is None else bool(payload.get("persist")),
            identity=payload.get("identity"))

    async def chat_delete(self, session_id: str) -> bool:
        """关闭会话：内存会话 + 对应工作记忆窗口一并释放（已落库流水不受影响）。"""
        hub = self._chat_hub()
        session = hub.get(session_id)
        identity = session.identity if session is not None else ""
        ok = hub.delete(session_id)
        registry = self.app.get("registry")
        if registry is not None and identity:
            registry.drop(f"webchat:{identity}:{session_id}")
        return ok

    async def chat_send(self, payload: dict[str, Any]) -> dict[str, Any]:
        hub = self._chat_hub()
        return await hub.submit(
            str(payload.get("session_id") or ""),
            str(payload.get("text") or ""),
            str(payload.get("user_name") or ""))

    async def chat_poll(self, session_id: str, after: int = 0) -> dict[str, Any]:
        return self._chat_hub().poll(session_id, after)

    async def chat_history(self, session_id: str) -> dict[str, Any]:
        return self._chat_hub().history(session_id)

    # ================================================================== 轨迹（MindLine §4.4）
    async def trace_list(self, after: int = 0, session: str = "",
                         limit: int = 300) -> list[dict[str, Any]]:
        """增量读取轨迹：优先内存环形缓冲（快），未装配时退回库查询。"""
        log = self.app.get("trace")
        if log is not None:
            return log.recent(after, session, limit)
        return await self.store.list_traces(after, session, limit)

    async def trace_clear(self) -> int:
        log = self.app.get("trace")
        if log is not None:
            log.clear()
        return await self.store.clear_traces()

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