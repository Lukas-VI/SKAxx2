# SKAxx2

一个可拓展的自主陪伴机器人：以「心跳 + 显式意识层 + 分层记忆」为核心，把 LLM 编排成**可控、可观测**的对话与主动行为链路。

默认通过 SnowLuma（OneBot v11）接入 QQ，自带零依赖 WebUI 管理后台。

## 核心特性

- **心跳驱动**：`bpm` 决定节拍；主动发言走「工作时间 / 频率 / 静默退避」三重闸 + 发送熔断，避免刷屏
- **显式意识层**：内心独白 → 决策 → 工具 → 反思，五步流程可见可控，无隐藏死循环
- **分层记忆**：SQLite + FTS5 trigram，中文子串召回；工作记忆定窗、会话摘要、长期记忆带时间衰减
- **多 LLM 提供商**：OpenAI 兼容协议，内置 Ollama / DeepSeek / OpenAI / 通义 / 智谱 / Kimi 预设，运行时可切换
- **权限三级模型**：后台角色 `owner/admin/viewer`，聊天用户等级 `blacklist<guest<member<admin<owner>`，工具级最小等级
- **在线聊天（多开 + 沙盒）**：WebUI 内直接对话，会话标签多开、独立上下文窗口，共享长期记忆/画像/权限；会话可设「沙盒」——照常推理但不落任何记忆库，便于边聊边调试、教智能体
- **动态上下文工程**：`ContextBuilder` 按模型窗口组装「人设 + 记忆召回 + 会话摘要 + 最近轮次」，占用接近上限自动压缩 + 记忆整理；`ContextPlan` 声明「要多少上下文」，为后续 llama.cpp 动态 KV cache 预留接口
- **MindLine 可观测**：三行泳道（输入 / 思考与回环 / 工具与输出）逐列回放模型的一次交互；轨迹既是日志、也落库，可增量拉取
- **零依赖 WebUI**：标准库 `http.server` + 原生单页，无构建、无第三方前端库，仅监听 `127.0.0.1`

## 目录结构

```
main.py              装配入口（全项目唯一 import 业务模块处）
start.bat|ps1|sh     各平台一键启动脚本（自动建 .venv 并装依赖）
config/
  config.yaml        声明式装配表：启用哪些渠道/工具/插件
  persona.yaml       人设分档（本地私有，已 gitignore，首次初始化自动生成）
  providers.yaml     本地 LLM 提供商实例与密钥（已 gitignore，首启自动生成）
  scheduler.yaml     定时任务（默认停用）
ska/
  kernel/            总线、事件、注册表、配置、数据包、会话上下文
  llm/               LLM 网关、结构化输出解析、提供商管理
  loop/              意识层（mind）、执行者（agent）、多会话登记表（sessions）
  context/           上下文工程：token 估算、预算、ContextBuilder/ContextPlan
  memory/            SQLite 记忆存储、召回、摘要
  tools/             内置工具（time / calculator / fetch / memory_ops；code 三件默认禁用）
  io/                渠道：cli、qq（OneBot v11）、webchat（在线聊天）
  heartbeat/         心跳循环与节流闸门
  plugins/           横切插件（违禁词过滤、定时唤醒）
  admin/             后台数据层（账号/设置/用量/用户/白名单）与权限策略、MindLine 轨迹（trace）
  web/               HTTP 服务、鉴权、ChatHub（在线聊天中枢）与静态管理界面
data/                运行期数据库（已 gitignore）
tests/               标准库 unittest 测试
```

## 快速开始

### 一键启动

首次运行会自动建 `.venv` 并装依赖，之后复用同一环境直接启动：

| 平台 | 命令 |
| --- | --- |
| Windows（cmd / 双击） | `start.bat` |
| Windows（PowerShell） | `powershell -ExecutionPolicy Bypass -File start.ps1` |
| Linux / macOS | `chmod +x start.sh && ./start.sh` |

### 手动启动

```powershell
pip install -r requirements.txt
python main.py
```

启动后浏览器打开 **http://127.0.0.1:5098**，进入「首次使用初始化」：设置管理员口令，可选地填 LLM 提供商与 QQ 适配器。

终端（CLI 渠道）也可直接对话，输入 `/quit` 或 `Ctrl+C` 退出。

## 接入 QQ（SnowLuma）

1. 启动 SnowLuma（OneBot v11 实现），记下 HTTP API 地址（默认 `http://127.0.0.1:3000`）与 accessToken
2. 在 SnowLuma 的 `httpClients` 中把事件上报地址配为 `http://127.0.0.1:5098/onebot/events`
3. 在后台「绑定 QQ 适配器」页填入 API 地址与 token，保存后热生效

发信走 `httpServers`（HTTP POST `send_group_msg` / `send_private_msg`），收信走事件上报，两侧都带 `Bearer` 鉴权。

## 配置优先级

同一小节按三层合并，后者覆盖前者：

1. 内置默认（代码常量）
2. `config/config.yaml`
3. 后台覆盖层（`data/admin.db` 的 `settings` 表，WebUI 改动写这里，不动 YAML）

> `config/persona.yaml`（人设文本）与 `config/providers.yaml`（LLM 密钥）属本地私有配置，已加入 `.gitignore`。
> 克隆仓库后**无需手动创建**：首次打开 WebUI 完成初始化时会自动生成（人设写内置通用模板、
> 提供商写入向导里填的内容，已存在则一律不改），随后可自行改写。

## 管理后台功能

运行状态 · **在线聊天** · **MindLine** · 绑定 QQ 适配器 · API 管理 · 用量管理 · 聊天用户管理 · 记忆管理 · 权限管理 · 插件拓展 · 智能体权限 · 可发布目标（白名单）· 设置

## 在线聊天 · 上下文工程 · MindLine

- **在线聊天**：后台「在线聊天」页可多开会话（标签栏一键新建 / 一键开「临时会话」）。多会话共享长期记忆、画像与权限，只有对话窗口与「是否落库」按会话隔离；沙盒会话照常推理，但不写 `messages`/`long`/画像/摘要。
- **上下文工程**：每个提供商可设 `context_window`（可在设置页覆盖）。`ContextBuilder` 把「人设 + 记忆召回 + 会话摘要 + 最近轮次」按预算装进窗口，占用达到 `compress_at`（默认 0.8）即滚动摘要 + 记忆整理并重建上下文。设计分工：**SKA 决定「要多少上下文」，后端决定「怎么省显存」**——`ContextPlan` 即前者，后续 llama.cpp 适配器消费它。
- **MindLine**：后台「MindLine」页按「列 = 一次交互、行 = 输入/思考/工具与输出」三行泳道回放，支持按会话 / 只看工具 / 只看错误 / 时间范围筛选，点开任意一格看完整 detail。轨迹同时写入 `ska.trace` 日志与 `data/admin.db` 的 `traces` 表。
- **编码智能体（可选）**：`ska/tools/code.py` 提供 `code_read` / `code_write` / `code_run` 三级工具，**默认全部禁用**；启用需在 `config.yaml` 的 `code` 小节打开并配置工作目录白名单，再到「智能体权限」页设为 admin 级。

## 测试

```powershell
python -m unittest discover -s tests -t .
```

## 依赖

必需三项：`pydantic`、`PyYAML`、`httpx`。明确不引入重型编排框架（LangGraph / OpenAI Agents SDK 等），薄 agent loop 更可控。

## 许可

[Apache License 2.0](LICENSE)
