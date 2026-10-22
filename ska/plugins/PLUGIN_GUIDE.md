# 插件实现指南（给 Agent 自己看）

本文件是 `ska/plugins/` 目录的**开发规范**。你要给自己扩展能力时，先读完本文件，
再动手写插件；写完用 `plugin_ops` 热装载，不必重启进程。

---

## 1. 目录与命名

| 规则 | 说明 |
|---|---|
| 位置 | 只能放 `ska/plugins/` 下的**单文件** Python 模块，如 `ska/plugins/my_tool.py` |
| 命名 | `^[A-Za-z_][A-Za-z0-9_]{0,63}$`；不能用中文、空格、连字符，不能以数字开头 |
| 忽略 | 以 `_` 开头的模块不会被当成插件（可放私有辅助模块） |
| 保留 | `manager.py` 是插件基础设施，**不能**当作插件读写/删除 |
| 内置 | `filter.py`（违禁词过滤）、`scheduler.py`（定时唤醒）禁止删除，只允许重载/启停 |
| 体积 | 源码上限 200000 字符（超出拒绝写入） |
| 依赖 | 只能用项目已有依赖（httpx / pydantic 等），**不要**新增第三方包 |

## 2. 最小可用模板

```python
"""一句话说明这个插件做什么。"""
from __future__ import annotations

import logging

from ..kernel.bus import EventBus       # 仅用于类型标注
from ..kernel.event import TOPIC_OUTPUT

logger = logging.getLogger(__name__)


class MyPlugin:
    def __init__(self, bus: EventBus) -> None:
        self.bus = bus

    def on_output(self, event) -> None:
        """事件处理器：同步/异步都支持（总线会 await 异步返回值）。"""
        logger.info("看到一条待发送消息：%r", getattr(event.payload, "content", ""))

    async def stop(self) -> None:
        """可选：卸载时调用，用来取消后台任务、关闭连接。"""
        return None


def install(bus: EventBus, cfg) -> MyPlugin:
    """模块级 install(bus, cfg) 是**唯一必需入口**，返回实例（或 None）。"""
    plugin = MyPlugin(bus)
    bus.subscribe(TOPIC_OUTPUT, plugin.on_output)
    logger.info("my_plugin 已装载")
    return plugin
```

要点：

- `install` 必须是**模块级函数**，签名固定为 `install(bus, cfg)`；写源码时管理器会用 AST 检查，
  没有它直接拒绝落盘。
- `bus` 是插件专属作用域（`_BusScope`）：你 `subscribe` 的每一项都会被记账，
  卸载/重载时自动摘除，**不需要**自己写 `unsubscribe`。
- `cfg` 是 `kernel.config.Config`：`cfg.get("plugins", [])` / `cfg.section("qq")` 这样取值。

## 3. 生命周期

```
写文件 → 语法+install 校验 → 导入模块 → install(bus, cfg) → （可选）contribute_tools(deps)
                                                        ↓
                                               热生效，工具表/订阅立即更新
卸载：stop()/uninstall() → 摘掉订阅 → 回收它贡献的工具 → 清 sys.modules 与 .pyc
```

- `stop()` 与 `uninstall()` 二者可选，可同步可异步；一个抛异常不会阻断其余回收。
- 重载 = 卸载 → 清模块缓存/字节码 → 重新导入 → 再 install。
- 因此**不要在 `install` 里做不可逆动作**（如写删数据库、改全局配置），否则重载会重复执行。

## 4. 可订阅的事件主题

| 常量 | 值 | 何时触发 | payload |
|---|---|---|---|
| `TOPIC_INPUT` | `input` | 收到用户消息（渠道归一化后） | `CoreInput` |
| `TOPIC_WAKE` | `wake` | 定时/外部主动唤醒 | `CoreInput(source="heartbeat")` |
| `TOPIC_CHANNEL_STATUS` | `channel_status` | QQ 渠道断线/重连 | `{"event": ..., "text": ...}` |
| `TOPIC_OUTPUT` | `output` | **发送前**的 `CoreOutput`（可原地改 `content`） | `CoreOutput` |
| `TOPIC_HEARTBEAT` | `heartbeat` | 每记一次心跳 | 心跳信息 |
| `TOPIC_TRACE` | `trace` | MindLine 轨迹 | 轨迹 dict |

想主动「叫醒」Agent，就 publish 一条 `TOPIC_WAKE`：

```python
from ..kernel.event import TOPIC_WAKE, Event
from ..kernel.packet import CoreInput

await bus.publish(Event(topic=TOPIC_WAKE, payload=CoreInput(
    source="heartbeat", task="my_task", event="my_plugin", content="该提醒他了")))
```

## 5. 贡献工具（让模型能调用）

插件可以给自己加工具，返回 `{工具名: Tool 实例}`：

```python
import os
from ..tools.base import Tool

class WeatherTool(Tool):
    description = "查询某城市天气（示例）。"
    parameters = {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "城市名"}},
        "required": ["city"],
    }

    async def run(self, city: str = "", **_) -> str:
        return f"{city}：晴，25℃（示例数据）"


class MyPlugin:
    def __init__(self, bus, cfg):
        self.bus, self.cfg = bus, cfg

    def contribute_tools(self, deps):
        return {"weather_demo": WeatherTool(deps)}


def install(bus, cfg):
    return MyPlugin(bus, cfg)
```

约束与注意：

- 工具名与已有工具冲突时，**以已有工具为准**，你的会被静默忽略（日志有 WARNING）。
- `run()` 必须 `async`，返回**字符串**（会回填进模型上下文，保持简洁）。
- 插件工具默认按 `member` 权限纳管；高危能力请提醒管理员到后台「智能体权限」页调高等级。
- 不要在 import 阶段做网络请求/读大文件：装载失败会连带整个插件不可用。

## 6. 用 `plugin_ops` 自我管理

| action | 作用 | 必填 |
|---|---|---|
| `list` | 列出插件状态（已启用/运行中/内置） | — |
| `read` | 读某插件源码（学现成写法） | `name` |
| `write` | 新建/覆写源码并**立即热装载** | `name`, `source`, 可选 `enable` |
| `reload` | 改完源码让它生效 | `name` |
| `enable` / `disable` | 热插拔（同时持久化到启用名单） | `name` |
| `delete` | 卸载并删除文件（内置插件禁止） | `name` |

推荐自举流程：

1. `plugin_ops(action="read", name="filter")` 看一个现成插件怎么接总线；
2. 想清楚触发点（事件订阅 or 工具）与失败时的降级行为；
3. `plugin_ops(action="write", name="my_plugin", source="...")`；
4. 用 `list` 确认 `loaded=true`；
5. 行为不对就改源码 → `reload` → 再看日志，不要反复 `write`。

也可以用 `code_write` 直接写 `ska/plugins/xxx.py`（该目录在 `code.allow_dirs` 白名单内），
但写完后仍需 `plugin_ops(action="reload")` 或 `enable` 才会生效。

## 7. 常见坑

- **热重载没变化**：确认改的是文件而不是只改了下游数据；管理器已清 `sys.modules` 和 `.pyc`，
  若仍不生效，检查 `install` 里是否有模块级缓存（如全局单例）。
- **订阅重复触发**：不要绕过 `bus.subscribe` 自己存 handler 到全局，卸载时不会被摘除。
  （除非你在 `stop()` 里显式 `unsubscribe`。）
- **卡住主循环**：事件处理器与 `run()` 都在主事件循环里执行；`time.sleep`、同步 requests、
  大循环会拖住整个机器人。用 `asyncio.sleep` 与 httpx 异步客户端。
- **插件之间互相 import**：`from .other_plugin import X` 会造成卸载残留；插件应保持自包含。
- **异常处理**：`install` 抛异常 → 装载失败（会回滚已注册订阅）；事件处理器抛异常 → 记日志并继续。
  对可预期的失败，返回可读文本比抛异常更利于模型自我纠正。

## 8. 自检清单

- [ ] 文件名合法、位于 `ska/plugins/`、单文件、无新增依赖
- [ ] 有模块级 `install(bus, cfg)`，返回实例
- [ ] 订阅都走 `bus.subscribe`（不要自己保存 handler 后忘记摘）
- [ ] 耗时操作用 `await`，不阻塞事件循环
- [ ] 有 `stop()` 时能幂等取消后台任务
- [ ] 贡献的工具 `run()` 是 async、返回字符串、描述清楚
- [ ] 装载后用 `plugin_ops(action="list")` 确认 `loaded=true`
