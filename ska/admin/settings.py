"""运行时设置（阶段二新增）：让「后台改的配置」在不重启进程的前提下生效。

为什么需要它：
    config.yaml 是「声明式装配表」，含大量注释，不适合被程序回写覆盖。
    因此后台的改动统一写进 admin.db 的 settings 表（按 section 存 JSON），
    由本类在内存中与 config.yaml 合并，供各组件同步读取。

    config.yaml（基底，含注释、可手改）
        +
    settings 表（后台覆盖层，JSON）
        ↓
    RuntimeSettings（内存快照；同步读取，热路径零开销）

覆盖层按「小节」组织，键名与 config.yaml 保持一致：
    app      → bpm / persona_profile / tools / plugins / channels（顶层与名单类）
    throttle → 主动发言三重闸
    memory   → 记忆参数
    agent    → 是否允许工具 / 最大步数
    qq       → SnowLuma 适配器连接参数
    web      → 后台监听参数（端口变更需重启）

一致性：本类只由 main.py 持有并被各组件引用；
后台写入时同时更新 DB 与内存快照，因此运行期读到的永远是最新值。
"""
from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

# 「app」小节里可以覆盖的顶层键（config.yaml 顶层 + 名单类）
APP_KEYS = ("bpm", "persona_profile", "tools", "plugins", "channels")

# 覆盖层允许的小节白名单（防止前端塞入任意键）
SECTIONS = ("app", "throttle", "memory", "agent", "qq", "web", "roles", "webchat",
            "dialog")  # 白名单（plugin:* 由 is_allowed_section 前缀放行）
# 插件自带设置项统一存成 "plugin:<插件名>" 小节：插件是运行期新增的，
# 无法预先列进 SECTIONS，故按前缀放行（仍限定前缀，不允许任意键）。
PLUGIN_SECTION_PREFIX = "plugin:"


def is_allowed_section(key: str) -> bool:
    """该小节名是否允许被后台覆盖层保存/读取。"""
    return key in SECTIONS or key.startswith(PLUGIN_SECTION_PREFIX)

# roles 小节默认值：聊天用户等级规则的全局设置（决策 D6）
DEFAULT_ROLES: dict[str, Any] = {
    "owner_user": "",       # 被视为 owner 等级的聊天用户 id（通常是管理员本人）
    "default_level": "member",   # 未登记用户/未指定时的默认等级
}

# agent 小节的默认值（config.yaml 里没有该小节时的兜底）
DEFAULT_AGENT: dict[str, Any] = {"allow_tools": True, "max_steps": 5}

# qq 小节的默认值（阶段二默认走 SnowLuma，见方案 §5.5）
# auto_start 系列（阶段三增强）：启用 QQ 且探测不到适配器时，按命令自动拉起 SnowLuma；
# 只在「本进程真的拉起过」时才在退出/停用时关闭它，不会误杀用户手动启动的实例。
DEFAULT_QQ: dict[str, Any] = {
    "enabled": False,
    "transport": "http",
    "api_base": "http://127.0.0.1:3000",
    "access_token": "",
    "event_path": "/onebot/events",
    "report_token": "",
    "self_id": 0,
    "default_group": "",
    "default_user": "",
    "reconnect_min": 1,
    "reconnect_max": 60,
    "auto_start": False,
    "snowluma_command": "",
    "snowluma_cwd": "",
    "snowluma_webui_port": 0,
    "startup_timeout": 60,
}

# web 小节默认值（D5：5098，避开 SnowLuma 的 5099）
DEFAULT_WEB: dict[str, Any] = {"enabled": True, "host": "127.0.0.1", "port": 5098,
                               "session_hours": 72}

# webchat 小节默认值（阶段三 §2.4 / 决策 E1、E2、E3）
#   identity / user —— 在线聊天代表的聊天用户身份（默认 owner，决定权限等级与记忆归属）
#   persist         —— 新建会话默认是否落库（false=沙盒，可单独在会话里改）
#   poll_ms         —— 前端轮询间隔（决策 E2：默认 1 秒）
#   compress_at     —— 上下文占用达到该比例即触发压缩（§3.3）
#   context_window  —— 本渠道覆盖模型窗口（0 = 跟随当前提供商，默认跟随）
DEFAULT_WEBCHAT: dict[str, Any] = {
    "enabled": True,
    "user": "owner",
    "identity": "owner",
    "persist": True,
    "poll_ms": 1000,
    "compress_at": 0.8,
    "context_window": 0,
}

# 有「内置默认值」的小节：合并顺序 = 内置默认 → config.yaml → 后台覆盖层。
# 这样即使 config.yaml 里没写这些小节，也能拿到可用默认值；
# 写了就作为初值（后台可再覆盖），语义最直观。
# 对话窗口（消息批处理）默认值：推理期间同一聊天堆积的消息合并成一批
DEFAULT_DIALOG: dict[str, Any] = {"batch_enabled": True, "batch_max_messages": 12}

_SECTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "agent": DEFAULT_AGENT, "qq": DEFAULT_QQ, "web": DEFAULT_WEB,
    "roles": DEFAULT_ROLES, "webchat": DEFAULT_WEBCHAT, "dialog": DEFAULT_DIALOG,
}


class RuntimeSettings:
    """配置基底（config.yaml）+ 后台覆盖层（settings 表）的合并视图。"""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg                          # kernel.config.Config
        self._over: dict[str, dict[str, Any]] = {}   # section → 覆盖 dict

    # ------------------------------------------------------------------ 载入 / 导出
    def load_sync(self, stored: dict[str, Any]) -> None:
        """用 settings 表的数据初始化覆盖层（启动阶段调用，此时无事件循环）。"""
        clean: dict[str, dict[str, Any]] = {}
        for key, value in (stored or {}).items():
            if is_allowed_section(key) and isinstance(value, dict):
                clean[key] = value
        self._over = clean
        self._apply_persona_override()          # 让 Config.build_system_prompt 也看到覆盖值

    def set_overrides(self, data: dict[str, Any]) -> None:
        """整体替换覆盖层（后台保存后调用，保证内存与 DB 同步）。"""
        self.load_sync(data)

    def overrides(self) -> dict[str, dict[str, Any]]:
        """覆盖层深拷贝（供后台「读取当前设置」与写回前合并）。"""
        return copy.deepcopy(self._over)

    # ------------------------------------------------------------------ 读取
    def get(self, key: str, default: Any = None) -> Any:
        """读顶层键：先看 app 覆盖层，再退回 config.yaml。"""
        app = self._over.get("app") or {}
        if key in app and app[key] is not None:
            return app[key]
        return self.cfg.get(key, default)

    def section(self, name: str) -> dict[str, Any]:
        """读某个小节：内置默认 ⊕ config.yaml 基底 ⊕ 后台覆盖层。"""
        base = dict(_SECTION_DEFAULTS.get(name, {}))   # 内置默认（可能为空）
        base.update(self.cfg.section(name))            # config.yaml 初值
        base.update(self._over.get(name) or {})        # 后台覆盖层（最高优先级）
        return base

    def names(self, key: str) -> list[str]:
        """读「名单类」配置（tools / plugins / channels）：覆盖层优先。"""
        value = self.get(key)
        return list(value) if isinstance(value, (list, tuple)) else []

    def persona_profile(self) -> str:
        """当前人设档名（local_full / cloud_safe）。"""
        override = (self._over.get("app") or {}).get("persona_profile")
        if override:
            return str(override)
        return str(self.cfg.cfg.get("persona", {}).get("profile", "local_full"))

    def agent(self) -> dict[str, Any]:
        return self.section("agent")

    def qq(self) -> dict[str, Any]:
        return self.section("qq")

    def web(self) -> dict[str, Any]:
        return self.section("web")

    # ------------------------------------------------------------------ 供后台展示
    def snapshot(self) -> dict[str, Any]:
        """返回当前「生效值」汇总，供后台「设置」页直接渲染。"""
        return {
            "app": {
                "bpm": self.get("bpm", 6),
                "persona_profile": self.persona_profile(),
                "tools": self.names("tools"),
                "plugins": self.names("plugins"),
                "channels": self.names("channels"),
            },
            "throttle": self.section("throttle"),
            "memory": self.section("memory"),
            "agent": self.agent(),
            "qq": self._mask_qq(self.qq()),
            "web": self.section("web"),
            "roles": self.section("roles"),
            "webchat": self.webchat(),
        }

    def roles(self) -> dict[str, Any]:
        return self.section("roles")

    def webchat(self) -> dict[str, Any]:
        """在线聊天小节（阶段三）：身份 / 默认落库 / 轮询 / 压缩阈值。"""
        return self.section("webchat")

    # ------------------------------------------------------------------ 内部
    def _apply_persona_override(self) -> None:
        """把覆盖的 persona_profile 写回 Config，使 build_system_prompt 生效。

        说明：Config.persona_profile() 读的是 cfg.cfg["persona"]["profile"]，
        这里直接改写该字段，避免为此改造 Config 的接口。
        """
        override = (self._over.get("app") or {}).get("persona_profile")
        if not override:
            return
        persona = self.cfg.cfg.setdefault("persona", {})
        if isinstance(persona, dict):
            persona["profile"] = str(override)

    @staticmethod
    def _mask_qq(qq: dict[str, Any]) -> dict[str, Any]:
        """QQ 小节对外展示时是否已配置密钥（不回明文，前端只显示占位）。"""
        out = dict(qq)
        out["has_token"] = bool(out.get("access_token"))
        return out