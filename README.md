# SKAxx2

一个可拓展的自主陪伴机器人：以「心跳 + 显式意识层 + 分层记忆」为核心，把 LLM 编排成**可控、可观测**的对话与主动行为链路。

默认通过 SnowLuma（OneBot v11）接入 QQ，自带零依赖 WebUI 管理后台。

## 核心特性

- **心跳驱动**：`bpm` 决定节拍；主动发言走「工作时间 / 频率 / 静默退避」三重闸 + 发送熔断，避免刷屏
- **显式意识层**：内心独白 → 决策 → 工具 → 反思，五步流程可见可控，无隐藏死循环
- **分层记忆**：SQLite + FTS5 trigram，中文子串召回；工作记忆定窗、会话摘要、长期记忆带时间衰减
- **多 LLM 提供商**：OpenAI 兼容协议，内置 Ollama / DeepSeek / OpenAI / 通义 / 智谱 / Kimi 预设，运行时可切换
- **权限三级模型**：后台角色 `owner/admin/viewer`，聊天用户等级 `blacklist<guest<member<admin<owner>`，工具级最小等级
- **零依赖 WebUI**：标准库 `http.server` + 原生单页，无构建、无第三方前端库，仅监听 `127.0.0.1`

## 目录结构

```
main.py              装配入口（全项目唯一 import 业务模块处）
start.bat|ps1|sh     各平台一键启动脚本（自动建 .venv 并装依赖）
config/
  config.yaml        声明式装配表：启用哪些渠道/工具/插件
  persona.yaml       人设分档（local_full / cloud_safe）
  providers.yaml     本地 LLM 提供商实例与密钥（已 gitignore，首启自动生成）
  scheduler.yaml     定时任务（默认停用）
ska/
  kernel/            总线、事件、注册表、配置、数据包、会话上下文
  llm/               LLM 网关、结构化输出解析、提供商管理
  loop/              意识层（mind）与执行者（agent）
  memory/            SQLite 记忆存储、召回、摘要
  tools/             内置工具（time / calculator / fetch / memory_ops）
  io/                渠道：cli、qq（OneBot v11）
  heartbeat/         心跳循环与节流闸门
  plugins/           横切插件（违禁词过滤、定时唤醒）
  admin/             后台数据层（账号/设置/用量/用户/白名单）与权限策略
  web/               HTTP 服务、鉴权与静态管理界面
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

## 管理后台功能

运行状态 · 绑定 QQ 适配器 · API 管理 · 用量管理 · 聊天用户管理 · 记忆管理 · 权限管理 · 插件拓展 · 智能体权限 · 可发布目标（白名单）· 设置

## 测试

```powershell
python -m unittest discover -s tests -t .
```

## 依赖

必需三项：`pydantic`、`PyYAML`、`httpx`。明确不引入重型编排框架（LangGraph / OpenAI Agents SDK 等），薄 agent loop 更可控。

## 许可

[Apache License 2.0](LICENSE)
