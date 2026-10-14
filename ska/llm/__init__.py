"""LLM 层：模型消息结构 + OpenAI 兼容网关 + 结构化输出解析。

对应老项目 server/LLM/Ollama_API.py 的升级：
    - requests（同步，阻塞事件循环）→ httpx.AsyncClient（真异步）；
    - 手写 JSON 拼 prompt → 原生 tools 参数 + response_format=json_object；
    - 无重试/无超时 → asyncio.timeout 包裹 + 一次结构化重试（见 structured.py）。
"""