"""SKAxx2 —— 可拓展、自觉型、陪伴聊天机器人。

分层原则：
    ska.kernel    只定义接口、数据契约与总线，不 import 任何业务模块；
    ska.llm       模型网关与结构化输出；
    ska.loop      自觉决策层（Mind）与执行者（AgentLoop）；
    ska.memory    分层长期记忆；
    ska.tools     工具（@tool 注册）；
    ska.io        渠道（@channel 注册）；
    ska.plugins   横切插件（install(bus, cfg)）；
    ska.heartbeat 心跳主循环。

装配只发生在 main.py：读配置 → 装插件 → 起渠道 → 跑心跳。
"""

__version__ = "0.1.0"