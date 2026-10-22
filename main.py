"""SKAxx2 装配入口。

这是全项目**唯一** import 业务模块的地方（方案硬约束）：
    读配置 → 建总线/记忆 → 建网关 → 建工具与渠道 → 装插件 → 建 Mind/Agent →
    起渠道 → 跑心跳 → 优雅退出。

设计取舍：
    * 内置工具是纯本地能力（time/calculator/fetch/memory_ops），不依赖外部模型；
    * 对话需要 LLM。本地没跑 Ollama 时，进程会照常启动，但回复会显示「模型不可用」，
      不会崩溃（LLM 调用全部有 asyncio.timeout + 异常兜底）。
"""
from __future__ import annotations

import asyncio
import importlib
import logging
import pkgutil
import signal
import sys
from typing import Any

from ska.heartbeat.beat import BeatLoop
from ska.kernel.bus import EventBus
from ska.kernel.config import Config
from ska.kernel.context import SessionContext
from ska.kernel.event import TOPIC_TRACE
from ska.loop.agent import AgentLoop
from ska.loop.mind import Mind
from ska.loop.sessions import SessionRegistry
from ska.memory.store import MemoryStore
from ska.memory.summarizer import Summarizer

logger = logging.getLogger("ska")


# --------------------------------------------------------------------------- 装配
def load_package(package: str) -> None:
    """导入一个包下的全部子模块，触发 @tool / @channel 等注册。

    这样「新增工具/渠道」只需新建文件 + 改 config.yaml，无需改 main.py。
    单个子模块导入失败只记警告（例如可选的 mcp_client 在未装 mcp 时）。
    """
    try:
        pkg = importlib.import_module(package)
    except ImportError:
        logger.exception("包导入失败：%s", package)
        return
    for info in pkgutil.walk_packages(getattr(pkg, "__path__", []), pkg.__name__ + "."):
        try:
            importlib.import_module(info.name)
        except Exception as e:  # noqa: BLE001 —— 可选模块缺失不应影响启动
            logger.warning("跳过子模块 %s：%s", info.name, e)


def build() -> dict[str, Any]:
    """按配置装配所有组件，返回一个 dict 容器（不启动）。"""
    cfg = Config()

    bus = EventBus()

    # 记忆层：SQLite 分层记忆（第三期核心）
    store: MemoryStore | None = None
    summarizer: Summarizer | None = None
    try:
        store = MemoryStore(cfg.memory.get("db_path", "data/ska.db"))
    except Exception:  # noqa: BLE001
        logger.exception("记忆库初始化失败，将以无记忆模式运行")

    # 管理库 + 运行时设置（阶段二）：后台改动写这里，不改 config.yaml（见方案 §5.6）
    admin_store: Any = None
    settings: Any = None
    try:
        from ska.admin.settings import RuntimeSettings
        from ska.admin.store import AdminStore
        admin_store = AdminStore()                              # data/admin.db
        settings = RuntimeSettings(cfg)
        # 启动阶段没有事件循环，必须用同步读取接口载入覆盖层
        settings.load_sync(admin_store.all_settings_sync())
    except Exception:  # noqa: BLE001
        logger.exception("管理库初始化失败，后台功能不可用（核心对话不受影响）")
        admin_store = settings = None

    # LLM 网关（阶段二改为「多提供商」：ProviderManager 持有唯一 gateway，
    # 首次启动会从 config.yaml 的 llm 小节迁移出默认提供商，见方案 §5.2）
    from ska.llm.providers import ProviderManager
    providers = ProviderManager(fallback=cfg.llm)
    gateway = providers.gateway
    if store is not None:
        summarizer = Summarizer(gateway, store)

    # 工具与渠道：先导入包触发注册，再按 config.yaml 名单实例化
    load_package("ska.tools")
    load_package("ska.io")

    # 阶段三：在线聊天中枢（ChatHub）+ 会话登记表 + MindLine 轨迹（§2/§3/§4）
    # 注意顺序：ChatHub 必须在渠道之前建好，才能经 ChannelDeps.extras 注入 webchat 渠道。
    from ska.context import ContextBuilder
    from ska.admin.trace import TraceLog
    from ska.web.chat import ChatHub

    # 上下文组装器（阶段三 §3.2）：窗口/系数做成回调，切换提供商后即时生效
    webchat_cfg = settings.webchat() if settings is not None else cfg.section("webchat")
    builder = ContextBuilder(window_fn=providers.context_window,
                             ratio_fn=providers.token_ratio,
                             compress_at=float(webchat_cfg.get("compress_at", 0.8) or 0.8))
    hub = ChatHub(bus,
                  default_identity=str(webchat_cfg.get("identity")
                                       or webchat_cfg.get("user") or "owner"),
                  default_persist=bool(webchat_cfg.get("persist", True)))
    registry = SessionRegistry(window=int(cfg.memory.get("working_window", 20)))
    # 轨迹落库（决策 E4：trace 是 log 的一部分）+ 内存环；订阅总线上的 TOPIC_TRACE
    trace = TraceLog(admin_store, buffer=500)          # admin_store 为 None 时只进日志
    trace.attach(bus)
    # ChatHub 也订阅轨迹：把上下文占用 token 回填到会话状态条
    bus.subscribe(TOPIC_TRACE, lambda ev: hub.on_trace(ev.payload))

    # ---- 图片 IO 与本地表情包库（阶段四）----
    # 视觉解析允许单独指定多模态端点（media.vision_model）；不指定就复用主网关
    # （主模型不支持图片时会自动降级为「未解析」，不会打断对话）。
    from ska.media import StickerBook, VisionDescriber
    media_cfg = settings.section("media") if settings is not None else cfg.section("media")
    vision_gateway = gateway
    if str(media_cfg.get("vision_model") or "").strip():
        from ska.llm.gateway import LLMGateway
        vision_gateway = LLMGateway({
            "id": "vision",
            "base_url": str(media_cfg.get("vision_base_url")
                            or cfg.llm.get("base_url", "")),
            "api_key": str(media_cfg.get("vision_api_key")
                           or cfg.llm.get("api_key", "")),
            "model": str(media_cfg.get("vision_model")),
            "max_tokens": int(media_cfg.get("vision_max_tokens", 400)),
            "timeout": float(cfg.llm.get("timeout", 60)),
        })
    vision = VisionDescriber(vision_gateway, media_cfg)
    # store 不可用时不做表情包库（无库可存），Mind 侧会自动跳过图片解析
    stickers = StickerBook(store, media_cfg, vision) if store is not None else None

    from ska.tools.base import ToolDeps, build_tools
    deps = ToolDeps(cfg=cfg, gateway=gateway, store=store, bus=bus)
    tools = build_tools(cfg.get("tools", []), deps)

    # MCP 外部工具（阶段三 §5 增强）：默认关闭。这里只读配置建管理器，
    # 真正的「拉起服务器 + 发现工具」放在运行阶段（需要事件循环），见 run()。
    from ska.mcp.bridge import MCPManager
    mcp = MCPManager.from_config(cfg)

    from ska.io.base import ChannelDeps, build_channels
    channels = build_channels(cfg.get("channels", ["cli"]),
                              ChannelDeps(bus=bus, cfg=cfg, settings=settings,
                                          extras={"chat_hub": hub}))

    # 人设 → system prompt（分档：local_full / cloud_safe）
    persona = cfg.build_system_prompt()

    # 权限策略（阶段二）：数据库缺失时置 None → 一切按阶段一行为（不裁权限）
    from ska.admin.policy import Policy
    policy = (Policy(admin_store, settings, tools)
              if admin_store is not None and settings is not None else None)

    # AgentLoop（执行者）+ Mind（决策者）；阶段三给二者注入总线以支持 MindLine 插桩
    agent = AgentLoop(gateway, tools, persona, bus=bus)
    mind = Mind(gateway, agent, store=store, summarizer=summarizer,
                cfg=cfg, persona_prompt=persona, policy=policy,
                bus=bus, builder=builder, stickers=stickers)

    # 心跳：单例 ctx 作为兜底，多会话走 registry 路由（阶段三 §2.2）
    ctx = SessionContext(window=int(cfg.memory.get("working_window", 20)))
    beat = BeatLoop(bus, cfg, mind, ctx, channels, settings=settings, policy=policy,
                    registry=registry)

    app: dict[str, Any] = {
        "cfg": cfg, "bus": bus, "store": store, "summarizer": summarizer,
        "gateway": gateway, "providers": providers, "tools": tools, "channels": channels,
        "agent": agent, "mind": mind, "ctx": ctx, "beat": beat,
        "plugins": [], "plugin_map": {},
        "admin_store": admin_store, "settings": settings, "policy": policy,
        "mcp": mcp, "tool_deps": deps,
        "service": None, "auth": None, "api": None, "web": None,
        # ---- 阶段三新增 ----
        "chat_hub": hub, "registry": registry, "trace": trace, "builder": builder,
        # ---- 阶段四新增：图片/表情包 ----
        "stickers": stickers, "vision": vision,
        "vision_gateway": vision_gateway if vision_gateway is not gateway else None,
    }

    # 插件运行时管理器（热装载/热重载/热卸载的唯一入口）：
    # 必须在 app 组装完之后建，因为它要读 app["tools"] / app["bus"] / app["plugin_map"]。
    # 同时挂进 tool_deps.extra，让 Agent 的 plugin_ops 工具能直接操作它（自举）。
    from ska.plugins.manager import PluginManager
    plugin_manager = PluginManager(app, settings=settings, store=admin_store)
    app["plugin_manager"] = plugin_manager
    deps.extra["plugin_manager"] = plugin_manager

    # 后台业务聚合 + HTTP 服务（阶段二 §5.3）
    if admin_store is not None and settings is not None:
        from ska.admin.service import AdminService
        from ska.web.api import API
        from ska.web.auth import AuthService
        from ska.web.server import WebServer
        service = AdminService(app=app, store=admin_store, memory=store,
                               settings=settings, providers=providers, policy=policy)
        gateway.on_usage = service.record_usage          # 每次 LLM 调用后把用量落库
        if app.get("vision_gateway") is not None:        # 独立视觉端点同样计入用量
            app["vision_gateway"].on_usage = service.record_usage
        web_cfg = settings.web()
        app["service"] = service
        app["auth"] = AuthService(admin_store,
                                  session_hours=int(web_cfg.get("session_hours", 72)))
        app["api"] = API(service, app["auth"])
        app["web"] = WebServer(app["api"], host=str(web_cfg.get("host", "127.0.0.1")),
                               port=int(web_cfg.get("port", 5098)))
    else:
        logger.warning("未启用 WebUI 管理后台（管理库不可用）")

    return app


async def install_plugins(app: dict[str, Any]) -> None:
    """按名装载插件（委托给 PluginManager，统一处理订阅/工具回收）。

    必须在事件循环内调用（run() 中 await）：scheduler 需要在循环里 create_task，
    插件卸载钩子也可能是协程。插件实例登记进 app["plugin_map"]（name → instance），
    后台可在运行期热卸载/热重载。
    """
    manager = app.get("plugin_manager")
    if manager is None:                       # 兜底：未走 build() 的调用方
        from ska.plugins.manager import PluginManager
        manager = PluginManager(app)
        app["plugin_manager"] = manager

    for name in manager.enabled_names():
        if name in app.get("plugin_map", {}):  # 幂等：已装的跳过
            continue
        try:
            await manager.install(name)
        except Exception as e:  # noqa: BLE001 —— 单个插件坏掉不拖垮启动
            logger.warning("插件 %s 装载失败：%s", name, e)


def _collect_plugin_tools(app: dict[str, Any], instance: Any) -> None:
    """把插件贡献的工具合入工具表（保留的兼容入口，委托给 PluginManager）。

    既有测试直接调用本函数；实现唯一化到 ska.plugins.manager.collect_plugin_tools，
    避免「两份逻辑、行为漂移」。
    """
    from ska.plugins.manager import collect_plugin_tools
    collect_plugin_tools(app, instance)


async def _start_mcp(app: dict[str, Any], mcp: Any) -> None:
    """后台连接 MCP 服务器，并把发现的工具合入装配好的工具表。

    单个服务器失败由 MCPManager 内部吞掉（只记日志），这里再兜一层异常，
    确保任何 MCP 问题都不会影响对话主链。
    """
    try:
        await mcp.connect_all()
        added = mcp.inject(app["tools"])
        if added:
            logger.info("MCP 工具已合入 AgentLoop：%d 个", added)
    except Exception:  # noqa: BLE001
        logger.exception("MCP 初始化异常（不影响主链）")


# --------------------------------------------------------------------------- 运行
async def run(app: dict[str, Any]) -> None:
    """启动渠道并进入心跳主循环。"""
    # 装插件（需在事件循环内，scheduler 依赖 create_task）
    await install_plugins(app)

    # MCP 外部工具：后台拉起服务器并把发现的工具合入工具表（同一 dict，AgentLoop 立即可见）。
    # 放后台是刻意的——服务器启动可能较慢/超时，不该拖住渠道与主循环的启动。
    mcp = app.get("mcp")
    if mcp is not None:
        mcp_task = asyncio.create_task(_start_mcp(app, mcp))
        app["mcp_task"] = mcp_task

    channels = app["channels"]
    # 先起渠道（接收任务），再进主循环
    for ch in channels.values():
        await ch.start()

    # 起后台 WebUI（阶段二）：把端口占用等问题局限在这里，不影响对话主链
    web = app.get("web")
    settings = app.get("settings")
    if web is not None and (settings is None or bool(settings.web().get("enabled", True))):
        try:
            web.start(asyncio.get_running_loop())
        except OSError as e:
            logger.error("WebUI 启动失败（端口 %s 可能被占用）：%s", web.port, e)

    # 信号处理：Ctrl+C → 优雅退出
    loop = asyncio.get_running_loop()
    beat = app["beat"]
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, beat.request_stop)
        except NotImplementedError:
            # Windows 部分信号不支持 add_signal_handler，用传统方式兜底
            signal.signal(sig, lambda s, f: beat.request_stop())

    try:
        await beat.run()
    finally:
        await shutdown(app)


async def shutdown(app: dict[str, Any]) -> None:
    """停止 WebUI、渠道、插件、记忆库与网关（幂等，可重复调用）。"""
    # 0) WebUI（先停 HTTP，避免退出时还有请求打进来）
    web = app.get("web")
    if web is not None:
        try:
            web.stop()
        except Exception:  # noqa: BLE001
            logger.exception("WebUI 停止异常")
    # 1) 渠道
    for ch in app["channels"].values():
        try:
            await ch.stop()
        except Exception:  # noqa: BLE001
            logger.exception("渠道停止异常：%s", getattr(ch, "name", ch))
    # 1.5) MCP：先取消可能仍在连接的后台任务，再关闭已建立的服务器连接
    mcp_task = app.get("mcp_task")
    if mcp_task is not None and not mcp_task.done():
        mcp_task.cancel()
    mcp = app.get("mcp")
    if mcp is not None:
        try:
            await mcp.stop()
        except Exception:  # noqa: BLE001
            logger.exception("MCP 关闭异常")
    # 2) 插件（优先按 plugin_map 遍历，覆盖运行期热装载的插件）
    plugin_map = app.get("plugin_map") or {}
    for plugin in list(plugin_map.values()) or app.get("plugins", []):
        stopper = getattr(plugin, "stop", None)
        if stopper:
            try:
                result = stopper()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001
                logger.exception("插件停止异常：%s", type(plugin).__name__)
    # 3) 记忆后台任务 + 连接
    summarizer = app.get("summarizer")
    if summarizer is not None:
        try:
            await summarizer.close()
        except Exception:  # noqa: BLE001
            logger.exception("摘要任务收尾异常")
    store = app.get("store")
    if store is not None:
        try:
            await store.close()
        except Exception:  # noqa: BLE001
            logger.exception("记忆库关闭异常")
    # 4) 网关连接池
    gateway = app.get("gateway")
    if gateway is not None:
        try:
            await gateway.aclose()
        except Exception:  # noqa: BLE001
            logger.exception("网关关闭异常")
    # 独立视觉网关（复用主网关时为 None，避免重复关闭）
    vision_gateway = app.get("vision_gateway")
    if vision_gateway is not None:
        try:
            await vision_gateway.aclose()
        except Exception:  # noqa: BLE001
            logger.exception("视觉网关关闭异常")
    # 5) 管理库（放最后：其他组件可能仍在读设置）
    admin_store = app.get("admin_store")
    if admin_store is not None:
        try:
            await admin_store.close()
        except Exception:  # noqa: BLE001
            logger.exception("管理库关闭异常")


# --------------------------------------------------------------------------- 入口
def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    app = build()
    try:
        asyncio.run(run(app))
    except KeyboardInterrupt:
        logger.info("收到 Ctrl+C，退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())