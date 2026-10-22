"""插件运行时管理器：装载 / 卸载 / 热重载 / 新建 / 删除。

为什么需要单独一层：
    原先「装载插件」的逻辑散落在 main.install_plugins 与 AdminService 里，
    两边各写一份 importlib，卸载时只调用 stop()，于是有两个隐患：
        ① 插件在 install 里 bus.subscribe 的订阅没人摘——热重载后同一事件被
           新旧两份实例各处理一次（幽灵重复）；
        ② 插件贡献的工具没人回收——卸载后 LLM 仍能看到并调用一个已经死掉的工具。
    本模块把「插件生命周期」收成唯一入口，并记录每个插件订阅了什么、贡献了什么，
    卸载/重载时逐项回收，从而支持真正的**热插拔**。

自举（Agent 给自己写插件）的边界：
    * 允许读写 `ska/plugins/` 下的单文件插件（写前做语法 + install 入口校验）；
    * 允许热重载/启停/删除自己写的插件；
    * 内置插件（filter / scheduler）**禁止删除**，但允许重载与启停——
      它们是内核默认横切能力，误删会让「违规词过滤」等兜底静默消失。
"""
from __future__ import annotations

import ast
import asyncio
import importlib
import inspect
import logging
import pkgutil
import re
import sys
from pathlib import Path
from typing import Any

from ..kernel.config import package_root

logger = logging.getLogger(__name__)

# 插件名合法字符：字母/数字/下划线，且不能以数字开头（与 Python 模块名一致）
_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

# 随仓库内置的插件：禁止删除（可重载/启停），避免把默认兜底能力删没了
BUILTIN_PLUGINS = frozenset({"filter", "scheduler"})

# 插件目录下的「基础设施模块」：它们不是插件，不能出现在清单里被启停/删除。
# （manager.py 自己就在这里；若允许删除，等于运行期把管理器源码删掉。）
INFRASTRUCTURE_MODULES = frozenset({"manager"})

# 单个插件源码长度上限：防止把大文件塞进 LLM 上下文或误写巨型文件
MAX_SOURCE_CHARS = 200_000


class PluginError(ValueError):
    """插件操作失败（名字非法 / 语法错误 / 导入失败 / 受保护）。

    继承 ValueError 的用意：Web API 对 ValueError 统一返回 400 + 原文，
    这样后台页面与 Agent 工具都能直接看到「为什么失败」，不必翻 500 日志。
    """


def _coerce_value(key: str, field: dict[str, Any], value: Any) -> Any:
    """把前端传来的值按 schema 归一（也顺手挡掉明显不合法的输入）。"""
    typ = str(field.get("type") or "text")
    if typ == "bool":
        return bool(value)
    if typ == "number":
        try:
            return float(value) if "." in str(value) else int(value)
        except (TypeError, ValueError) as e:
            raise PluginError(f"设置项 {key} 需要数字") from e
    if typ == "targets":
        # 目标统一存成 "渠道|类型|ID"，例如 "qq|group|123456"：
        # 前端多选框直接给出这个字符串，插件侧 split 后就能构造 CoreOutput。
        if not isinstance(value, (list, tuple)):
            raise PluginError(f"设置项 {key} 需要目标列表")
        out: list[str] = []
        for item in list(value)[:20]:               # 上限 20 个，防一次刷一屏
            text = str(item or "").strip()
            parts = text.split("|")
            if len(parts) != 3 or not all(p.strip() for p in parts):
                raise PluginError(
                    f"设置项 {key} 的目标格式应为 渠道|group/user|ID：{text}")
            out.append("|".join(p.strip() for p in parts))
        return out
    if typ == "select":
        options = [str(o.get("value") if isinstance(o, dict) else o)
                   for o in field.get("options") or []]
        text = str(value or "")
        if options and text not in options:
            raise PluginError(f"设置项 {key} 只能是：{', '.join(options)}")
        return text
    return str(value or "")       # text / textarea


def plugins_dir() -> Path:
    """插件目录的绝对路径（项目根 / ska / plugins）。"""
    return Path(package_root()) / "ska" / "plugins"


def collect_plugin_tools(app: dict[str, Any], instance: Any) -> list[str]:
    """把插件贡献的工具合入工具表，返回实际新增的工具名列表。

    约定：插件实例可选实现 `contribute_tools(deps) -> {工具名: Tool}`。
    与 MCPManager.inject 同理：合入的是同一张 tools dict，因此 AgentLoop 立即可见、
    Policy 自动纳管；名字冲突时以既有工具为准（不覆盖内置/已装工具）。
    """
    contributor = getattr(instance, "contribute_tools", None)
    if not callable(contributor):
        return []
    try:
        produced = contributor(app.get("tool_deps"))
    except Exception:  # noqa: BLE001 —— 插件出错不能拖垮装配
        logger.exception("插件贡献工具失败：%s", type(instance).__name__)
        return []
    if not isinstance(produced, dict):
        return []
    target = app["tools"]
    added: list[str] = []
    for name, tool in produced.items():
        if name in target:
            logger.warning("插件工具名冲突，忽略：%s", name)
            continue
        target[name] = tool
        added.append(name)
    if added:
        logger.info("插件 %s 贡献了 %d 个工具", type(instance).__name__, len(added))
    return added


class _BusScope:
    """插件专属的总线视图：记录它注册的订阅，卸载时统一摘除。

    其余属性/方法（publish / emit / qsize / wakeup …）通过 __getattr__ 透传给真总线，
    所以插件代码无需感知这层包装，既有写法继续可用。
    """

    def __init__(self, bus: Any) -> None:
        self._bus = bus
        self._subs: list[tuple[str, Any]] = []

    def subscribe(self, topic: str, handler: Any) -> None:
        self._bus.subscribe(topic, handler)
        self._subs.append((topic, handler))

    def unsubscribe(self, topic: str, handler: Any) -> bool:
        return self._bus.unsubscribe(topic, handler)

    async def publish(self, event: Any) -> None:
        await self._bus.publish(event)

    async def emit(self, topic: str, payload: Any = None) -> None:
        await self._bus.emit(topic, payload)

    def close(self) -> None:
        """摘掉本插件注册过的全部订阅（幂等）。"""
        for topic, handler in self._subs:
            try:
                self._bus.unsubscribe(topic, handler)
            except Exception:  # noqa: BLE001 —— 卸载失败不应阻断后续回收
                logger.debug("取消订阅失败 topic=%s", topic, exc_info=True)
        self._subs.clear()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._bus, item)


class PluginManager:
    """插件生命周期唯一入口（main 装配、后台页面、Agent 工具共用）。"""

    def __init__(self, app: dict[str, Any], *, settings: Any = None,
                 store: Any = None) -> None:
        self.app = app
        self.bus = app["bus"]
        self.cfg = app["cfg"]
        self.settings = settings if settings is not None else app.get("settings")
        self.store = store if store is not None else app.get("admin_store")
        # 插件名 → 它贡献的工具名（卸载/重载时按此回收）
        self._tools: dict[str, list[str]] = {}
        # 插件名 → 它的总线作用域（用于摘订阅）
        self._scopes: dict[str, _BusScope] = {}

    # ------------------------------------------------------------------ 查询
    @staticmethod
    def available() -> list[str]:
        """枚举插件目录下的可用模块名（排除内部模块与基础设施模块 manager）。"""
        pkg = importlib.import_module("ska.plugins")
        return sorted(info.name.rsplit(".", 1)[-1]
                      for info in pkgutil.iter_modules(pkg.__path__)
                      if not info.name.startswith("_")
                      and info.name not in INFRASTRUCTURE_MODULES)

    def loaded(self) -> set[str]:
        return set(self.app.get("plugin_map", {}))

    def enabled_names(self) -> list[str]:
        """配置中「启用」的插件名单（后台覆盖层优先，其次 config.yaml）。"""
        if self.settings is not None:
            return list(self.settings.names("plugins"))
        return list(self.cfg.get("plugins", []))

    def describe(self) -> list[dict[str, Any]]:
        """给后台/Agent 看的清单：可用、是否启用、是否已装载、是否受保护。

        has_settings：插件是否声明了模块级 SETTINGS（有的话后台会显示「设置」按钮）。
        这里顺带 import 一下插件模块来读 SETTINGS；插件模块只有定义、无副作用，
        导入失败（语法错等）也不应拖垮清单，按「没有设置项」处理。
        """
        enabled = set(self.enabled_names())
        loaded = self.loaded()
        return [{"name": name, "enabled": name in enabled, "loaded": name in loaded,
                 "builtin": name in BUILTIN_PLUGINS,
                 "has_settings": bool(self.settings_schema(name))}
                for name in self.available()]

    # ------------------------------------------------------------------ 插件设置项
    def settings_schema(self, name: str) -> dict[str, Any]:
        """读取插件声明的设置项 schema（模块级 SETTINGS；没有则返回 {}）。

        schema 约定（保持极简，够前端生成表单即可）：
            {
              "title": "定时任务",
              "fields": [
                {"key": "targets", "label": "消息通道", "type": "targets",
                 "multiple": True, "default": [], "help": "..."},
              ]
            }
        type 支持：text / textarea / number / bool / select / targets。
        """
        self._check_name(name)
        try:
            module = importlib.import_module(f"ska.plugins.{name}")
        except Exception:  # noqa: BLE001 —— 插件本身有问题时不要影响清单/设置读取
            logger.debug("读取插件设置 schema 失败：%s", name, exc_info=True)
            return {}
        schema = getattr(module, "SETTINGS", None)
        if callable(schema):
            schema = schema()
        if not isinstance(schema, dict):
            return {}
        fields = schema.get("fields")
        return schema if isinstance(fields, list) else {}

    @staticmethod
    def default_settings(schema: dict[str, Any]) -> dict[str, Any]:
        """按 schema 取出默认值（新增字段后老配置也能拿到合理初值）。"""
        out: dict[str, Any] = {}
        for field in (schema or {}).get("fields") or []:
            if isinstance(field, dict) and str(field.get("key") or ""):
                out[str(field["key"])] = field.get("default")
        return out

    def validate_settings(self, name: str, values: dict[str, Any]) -> dict[str, Any]:
        """按 schema 校验并归一设置值（未知键直接拒绝，避免写库塞垃圾）。"""
        schema = self.settings_schema(name)
        if not schema:
            raise PluginError(f"插件 {name} 没有声明设置项")
        if not isinstance(values, dict):
            raise PluginError("设置值必须是对象")
        fields = {str(f.get("key")): f for f in schema.get("fields") or []
                  if isinstance(f, dict) and f.get("key")}
        unknown = [k for k in values if k not in fields]
        if unknown:
            raise PluginError(f"未知设置项：{', '.join(unknown)}")
        clean: dict[str, Any] = {}
        for key, field in fields.items():
            if key not in values:
                clean[key] = field.get("default")
                continue
            clean[key] = _coerce_value(key, field, values[key])
        return clean

    # ------------------------------------------------------------------ 装载 / 卸载
    async def install(self, name: str) -> Any:
        """导入并 install 一个插件；重复装载直接返回既有实例（幂等）。"""
        self._check_name(name)
        plugin_map = self.app.setdefault("plugin_map", {})
        if name in plugin_map:
            return plugin_map[name]

        module_name = f"ska.plugins.{name}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:  # noqa: BLE001 —— 把导入错误翻译成可读信息
            raise PluginError(f"插件 {name} 导入失败：{e}") from e

        install_fn = getattr(module, "install", None)
        if not callable(install_fn):
            raise PluginError(f"插件 {name} 没有 install(bus, cfg) 入口")

        scope = _BusScope(self.bus)       # 用包装总线调 install，便于记账
        # 插件如果声明了第三个参数 settings，就把运行时设置传进去（插件设置项的数据源）。
        # 用签名判断而不是 try/except TypeError：避免把插件内部真正的 TypeError 吞掉。
        kwargs: dict[str, Any] = {}
        try:
            if "settings" in inspect.signature(install_fn).parameters:
                kwargs["settings"] = self.settings
        except (TypeError, ValueError):
            pass
        try:
            instance = install_fn(scope, self.cfg, **kwargs)
        except Exception as e:  # noqa: BLE001
            scope.close()                 # 半途失败也要摘掉已注册的订阅
            raise PluginError(f"插件 {name} 装载失败：{e}") from e

        plugin_map[name] = instance
        self._scopes[name] = scope
        if instance is not None:
            self.app.setdefault("plugins", []).append(instance)
        self._tools[name] = collect_plugin_tools(self.app, instance)
        logger.info("插件已装载：%s（贡献工具 %d 个）", name, len(self._tools[name]))
        return instance

    async def uninstall(self, name: str) -> bool:
        """卸载插件：调 stop() → 摘订阅 → 回收工具。返回是否真的卸载了。"""
        instance = self.app.setdefault("plugin_map", {}).pop(name, None)
        scope = self._scopes.pop(name, None)
        if instance is None and scope is None:
            return False

        for hook in ("stop", "uninstall"):
            fn = getattr(instance, hook, None)
            if not callable(fn):
                continue
            try:
                result = fn()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001 —— 一个钩子失败不应阻断回收
                logger.exception("插件卸载钩子异常：%s.%s", name, hook)

        if scope is not None:
            scope.close()

        # 回收工具：只删「该插件当初新增的」，不误删同名内置工具
        tools = self.app.get("tools", {})
        for tool_name in self._tools.pop(name, []):
            tools.pop(tool_name, None)

        # 从插件实例列表移除（main.shutdown 的兜底遍历用）
        catalog = self.app.get("plugins", [])
        if instance is not None:
            for i, existing in enumerate(catalog):
                if existing is instance:
                    catalog.pop(i)
                    break
        logger.info("插件已卸载：%s", name)
        return True

    async def reload(self, name: str) -> Any:
        """热重载：卸载 → 清空模块缓存 → 重新导入并装载。

        必须清 sys.modules 并 invalidate_caches：Python 的 import 有缓存，
        只改文件不 reload 的话，装载到的仍是旧代码（这是「热重载不生效」的常见坑）。
        """
        self._check_name(name)
        await self.uninstall(name)
        self._purge_modules(name)
        importlib.invalidate_caches()
        return await self.install(name)

    async def set_enabled(self, name: str, enabled: bool) -> None:
        """启用/停用并持久化名单；随后调用方负责 install/uninstall。

        支持有后台库与无后台库两种运行形态：前者写 settings 覆盖层，
        后者只改内存名单（下次重启以 config.yaml 为准）。
        """
        names = self.enabled_names()
        if enabled and name not in names:
            names.append(name)
        if not enabled and name in names:
            names.remove(name)
        if self.settings is not None and self.store is not None:
            over = self.settings.overrides()
            over.setdefault("app", {})["plugins"] = names
            await self.store.save_settings(over)
            self.settings.set_overrides(over)

    async def toggle(self, name: str, enabled: bool) -> dict[str, Any]:
        """后台「启用/停用」按钮的实现：先校验，再改名单 + 立刻热装/热卸。"""
        self._check_name(name)
        if name not in self.available():
            raise PluginError(f"未知插件：{name}")
        await self.set_enabled(name, enabled)
        if enabled:
            await self.install(name)
        else:
            await self.uninstall(name)
        return {"name": name, "enabled": enabled, "loaded": name in self.loaded()}

    # ------------------------------------------------------------------ 源码读写
    def path_for(self, name: str) -> Path:
        """插件文件路径（同时做名字校验，杜绝 ../ 越界写）。"""
        self._check_name(name)
        # 基础设施模块（manager.py 自己）不允许被当作插件读写/删除，
        # 否则 Agent 一次误写就能把插件管理器本身改坏。
        if name in INFRASTRUCTURE_MODULES:
            raise PluginError(f"{name} 是插件基础设施模块，不能作为插件读写")
        return plugins_dir() / f"{name}.py"

    def read_source(self, name: str) -> str:
        path = self.path_for(name)
        if not path.exists():
            raise PluginError(f"插件不存在：{name}")
        return path.read_text(encoding="utf-8")

    def write_source(self, name: str, source: str) -> Path:
        """新建/覆写插件源码；写前做语法与入口校验，失败不落盘。"""
        path = self.path_for(name)
        if not isinstance(source, str) or not source.strip():
            raise PluginError("源码不能为空")
        if len(source) > MAX_SOURCE_CHARS:
            raise PluginError(f"源码过长（>{MAX_SOURCE_CHARS} 字符）")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as e:
            raise PluginError(f"语法错误：第 {e.lineno} 行 {e.msg}") from e
        # 必须有模块级 install(bus, cfg)：否则装载时一定失败，不如写之前就拒绝
        if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == "install" for node in tree.body):
            raise PluginError("插件必须定义模块级 install(bus, cfg) 函数")
        path.write_text(source, encoding="utf-8")
        logger.info("插件源码已写入：%s（%d 字符）", path, len(source))
        return path

    async def delete(self, name: str, *, remove_file: bool = True) -> bool:
        """卸载并删除插件文件（内置插件受保护，只允许卸载不允许删文件）。"""
        self._check_name(name)
        if remove_file and name in BUILTIN_PLUGINS:
            raise PluginError(f"{name} 是内置插件，禁止删除（可停用或重载）")
        path = self.path_for(name)
        removed = await self.uninstall(name)
        await self.set_enabled(name, False)
        if remove_file and path.exists():
            path.unlink()
            removed = True
        self._purge_modules(name)         # 连字节码缓存一起清（否则删了还能被导入）
        importlib.invalidate_caches()
        logger.info("插件已删除：%s", name)
        return removed

    async def create(self, name: str, source: str, *, enabled: bool = True) -> dict[str, Any]:
        """一步完成「写源码 → 加入名单 → 热装载」，供 Agent 自举使用。"""
        self.write_source(name, source)
        if not enabled:
            # 写入但要求停用：已装载的要立刻卸掉，否则会出现
            # 「配置里停用、运行中却还活着」的不一致状态。
            await self.set_enabled(name, False)
            await self.uninstall(name)
        else:
            await self.set_enabled(name, True)
            if name in self.loaded():
                await self.reload(name)
            else:
                await self.install(name)
        return {"name": name, "enabled": name in self.enabled_names(),
                "loaded": name in self.loaded()}

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _check_name(name: str) -> None:
        if not _NAME_RE.match(str(name or "")):
            raise PluginError("插件名只能包含字母/数字/下划线，且不能以数字开头")

    @staticmethod
    def _purge_modules(name: str) -> None:
        """清掉该插件在 sys.modules 中的缓存（含其子模块）与字节码文件。

        为什么连 .pyc 一起删：Python 判断字节码缓存是否新鲜，靠的是
        「源文件 mtime + 文件大小」。若两次写入发生在同一秒内、且长度不变
        （例如只把 `return "ok"` 改成 `return "v2"`），旧 pyc 会被继续复用，
        表现为「热重载后行为还是旧的」——这是热重载最隐蔽的一个坑。
        """
        prefix = f"ska.plugins.{name}"
        for module_name in [m for m in sys.modules
                            if m == prefix or m.startswith(prefix + ".")]:
            sys.modules.pop(module_name, None)
        try:
            cache = plugins_dir() / "__pycache__"
            for pyc in cache.glob(f"{name}.*.pyc"):
                pyc.unlink(missing_ok=True)
        except OSError:  # 缓存目录不可写/不存在都不影响正常导入
            logger.debug("清理插件字节码失败：%s", name, exc_info=True)
