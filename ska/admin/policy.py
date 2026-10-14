"""权限策略 Policy（阶段二新增，见方案 §5.7）。

三级权限模型（决策 D6）：
    聊天用户等级（低→高）：blacklist < guest < member < admin < owner
        * blacklist → 完全忽略（不回话、不进记忆）
        * guest     → 允许对话，但工具受限
        * member    → 默认等级
        * admin/owner → 可用全部工具
    工具级：每个工具可设「最小可用等级」与启用开关（tool_permissions 表）
    主动发言白名单：publish_targets 表（决策 D7：未配置时退回阶段一行为）

设计约束（向后兼容）：
    * Policy 可以是 None（未注入）——此时 Mind/Beat 行为与阶段一完全一致；
    * admin.db 为空（无 chat_users / 无 tool_permissions）时，等级一律为默认值，
      工具一律放行，白名单一律视为「未配置」。
"""
from __future__ import annotations

import logging
from typing import Any

from ..kernel.packet import CoreInput

logger = logging.getLogger(__name__)

# 聊天用户等级，按权限升序排列（索引越大权限越高）
LEVELS = ("blacklist", "guest", "member", "admin", "owner")
_LEVEL_RANK = {name: i for i, name in enumerate(LEVELS)}
DEFAULT_LEVEL = "member"


def level_rank(level: str) -> int:
    """把等级名转成可比较的序号；未知等级按最低处理（安全默认）。"""
    return _LEVEL_RANK.get(str(level or ""), 0)


class Policy:
    """权限判定：注入 Mind（对话路径）与 BeatLoop（主动路径）。"""

    def __init__(self, store: Any, settings: Any, tools: dict[str, Any] | None = None) -> None:
        self.store = store                  # admin.store.AdminStore
        self.settings = settings            # admin.settings.RuntimeSettings
        self.tools = tools or {}            # 已装配的工具（name → Tool），用于裁剪

    # ------------------------------------------------------------------ 用户等级
    async def level_of(self, channel: str, user_id: str) -> str:
        """判定某个聊天用户的等级。

        优先级：配置里指定的 owner_user > chat_users 表记录 > 默认等级。
        """
        roles = self.settings.roles()
        owner = str(roles.get("owner_user") or "").strip()
        if user_id and owner and str(user_id) == owner:
            return "owner"
        row = await self.store.get_chat_user(channel, user_id)
        if row and str(row.get("level") or ""):
            return str(row["level"])
        return str(roles.get("default_level") or DEFAULT_LEVEL)

    async def note_user(self, packet: CoreInput) -> None:
        """记录「出现过的聊天用户」，供后台「聊天用户管理」页展示。"""
        if packet.is_user_message() and packet.user:
            await self.store.touch_chat_user(packet.channel, packet.user, packet.user_name)

    async def is_blacklisted(self, channel: str, user_id: str) -> bool:
        row = await self.store.get_chat_user(channel, user_id)
        return bool(row and row.get("blacklisted"))

    # ------------------------------------------------------------------ 对话放行
    async def should_respond(self, packet: CoreInput) -> bool:
        """是否应该处理这条输入（拉黑用户直接忽略）。"""
        if not packet.is_user_message() or not packet.user:
            return True                      # 非用户消息（心跳/系统事件）不受此闸限制
        if await self.is_blacklisted(packet.channel, packet.user):
            logger.info("用户 %s 已被拉黑，忽略其消息", packet.user)
            return False
        return True

    # ------------------------------------------------------------------ 工具裁剪
    async def allowed_tools(self, level: str) -> set[str]:
        """按等级算出可用工具名集合（供 AgentLoop 过滤 tools 声明）。"""
        if not bool(self.settings.agent().get("allow_tools", True)):
            return set()                     # 全局关闭工具
        perms = {p["tool"]: p for p in await self.store.list_tool_permissions()}
        rank = level_rank(level)
        allowed: set[str] = set()
        for name in self.tools:
            perm = perms.get(name)
            if perm is None:
                allowed.add(name)            # 未单独配置 → 默认放行（阶段一行为）
                continue
            if bool(perm.get("enabled", 1)) and level_rank(perm.get("min_level", DEFAULT_LEVEL)) <= rank:
                allowed.add(name)
        return allowed

    async def tools_for(self, packet: CoreInput) -> set[str] | None:
        """给某条输入的可用工具集；None 表示「不裁剪」（非用户消息路径）。"""
        if not packet.is_user_message() or not packet.user:
            return None
        level = await self.level_of(packet.channel, packet.user)
        return await self.allowed_tools(level)

    # ------------------------------------------------------------------ 主动发言白名单
    async def targets(self, channel: str = "") -> list[dict[str, Any]]:
        """列出启用的可发布目标（按渠道筛选）。"""
        rows = await self.store.list_targets()
        return [t for t in rows
                if bool(t.get("enabled")) and (not channel or t.get("channel") == channel)]

    async def can_publish(self, channel: str, target_type: str, target_id: str) -> bool:
        """该目标是否允许机器人主动发言（D7：未配置白名单 → 放行，保持阶段一行为）。"""
        rows = await self.store.list_targets()
        scoped = [t for t in rows if not channel or t.get("channel") == channel]
        if not scoped:
            return True                      # 未配置 → 退回默认行为
        for t in scoped:
            if (t.get("target_type") == target_type
                    and str(t.get("target_id")) == str(target_id)):
                return bool(t.get("enabled"))
        return False                         # 配了白名单但不在其中 → 不允许