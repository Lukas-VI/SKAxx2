"""OpenAI 兼容 LLM 网关（httpx 异步）。

支持任意遵循 OpenAI /chat/completions 协议的后端：
    - Ollama：       base_url = http://localhost:11434/v1，api_key 随意填
    - DeepSeek：     base_url = https://api.deepseek.com/v1，api_key 填真实 key
    切换后端只需改 config.yaml 的 llm 小节，不用动代码。

要点：
    * 复用同一个 AsyncClient（连接池），进程结束前调用 aclose()；
    * 每次请求用 asyncio.timeout 兜底，避免模型卡死拖住心跳循环；
    * 原生支持 tools（函数调用）与 response_format=json_object。
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Sequence

import httpx

from .models import ChatMessage, LLMResponse, ToolCall

logger = logging.getLogger(__name__)

# 用量回调签名：provider_id, 实际模型名, 用量 dict, 任务标签
# 返回值可以是 None（同步）或 coroutine（异步）；网关两种都兼容。
UsageCallback = Callable[[str, str, dict[str, Any], str], Any]

# 中转「输出被截断」报错体里的特征串（命中即视为可重试的预算不足）。
# 各家中转措辞不一，这里只匹配可确认语义的片段，避免把普通错误误判为截断。
_TRUNCATION_MARKERS = (
    "structured_output_failed",     # 部分中转的结构化输出失败类型
    "finish_reason=length",         # 文本形式
    '"finish_reason": "length"',    # JSON 形式（带空格）
    '"finish_reason":"length"',     # JSON 形式（无空格）
    "output budget",                # "increase the output budget"
)


def _looks_truncated(text: str) -> bool:
    """判断一段报错文本是否表示「输出被截断 / 预算不足」。"""
    low = (text or "").lower()
    return any(marker in low for marker in _TRUNCATION_MARKERS)


class LLMError(RuntimeError):
    """LLM 调用失败（网络/超时/协议错误）。"""


class LLMTruncatedError(LLMError):
    """输出被 max_tokens 截断（finish_reason=length）。

    单独成类的原因：这不是「调用失败」，而是**预算不足**——上层（structured_chat）
    可以对它做「按 2 倍预算自动重试」，而普通 LLMError（网络/鉴权错）重试无意义。
    部分 OpenAI 兼容中转会把截断直接当硬错误返回（如 HTTP 502 structured_output_failed），
    本类同时覆盖「200+finish_reason=length」与「中转报错体含截断特征」两种情形。
    """


class LLMGateway:
    """异步 LLM 网关。

    阶段二新增（多提供商支持，见方案 §5.2）：
        * provider_id：标识「当前用的是哪个提供商」，供用量记录打标签；
        * reconfigure(cfg)：运行时不换对象就切换 base_url/key/model（惰性重建连接）；
        * on_usage：每次调用结束后把用量交给上层（AdminStore 落库）。
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._apply(cfg)                        # 统一从 cfg 载入字段（reconfigure 复用）
        self._client: httpx.AsyncClient | None = None
        self._retired: list[httpx.AsyncClient] = []  # 被 reconfigure 换下的旧连接池，待关闭
        self.on_usage: UsageCallback | None = None   # 用量回调（默认关闭 → 阶段一行为不变）

    # ------------------------------------------------------------------ 配置
    def _apply(self, cfg: dict[str, Any]) -> None:
        """把配置 dict 的字段写入实例（__init__ 与 reconfigure 共用）。"""
        # 去掉结尾斜杠，拼 URL 时避免出现双斜杠
        self.base_url = str(cfg.get("base_url", "http://localhost:8080/v1")).rstrip("/")
        self.api_key = str(cfg.get("api_key", ""))
        self.model = str(cfg.get("model", "qwen2.5:7b"))
        self.default_temperature = float(cfg.get("temperature", 0.8))
        self.max_tokens = int(cfg.get("max_tokens", 1024))
        self.timeout = float(cfg.get("timeout", 6000))
        self.retry = int(cfg.get("retry", 1))   # 供 structured 层读取
        self.provider_id = str(cfg.get("id", "") or cfg.get("provider_id", ""))

    def reconfigure(self, cfg: dict[str, Any]) -> None:
        """运行时不换网关对象就切换提供商（WebUI「切换/编辑」调用）。

        关键点：旧连接池带着旧 Authorization 头，必须丢弃重建，
        否则「改完 key 不生效」。这里把旧 client 暂存到 _retired，
        等 aclose() 时统一关闭，避免连接泄漏（也避免在同步方法里 await）。
        """
        self._apply(cfg)
        if self._client is not None:
            self._retired.append(self._client)
            self._client = None                 # 下次调用惰性重建，新 key 立即生效

    # ------------------------------------------------------------------ 生命周期
    def _ensure_client(self) -> httpx.AsyncClient:
        """惰性创建并复用 AsyncClient。

        httpx 自带的 timeout 略大于 asyncio 侧的 timeout，让 asyncio 先触发，
        这样能拿到明确的超时异常。
        """
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                # 连接阶段给一个较短、独立的超时：地址不可达/端口不通时快速失败，
                # 报错措辞与「读取慢」区分开。读取（生成）阶段给 timeout+5，
                # 让外层 asyncio.timeout 先于 httpx 触发，从而统一走「整体超时」分支。
                # 思考模型生成慢时，主要卡在读取阶段。
                timeout=httpx.Timeout(self.timeout + 5.0,
                                      connect=min(10.0, self.timeout)),
            )
        return self._client

    async def aclose(self) -> None:
        """关闭底层连接池（含被 reconfigure 换下的旧连接池）。进程退出时调用。"""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        for stale in self._retired:            # 补齐切换提供商时遗留的连接池
            try:
                await stale.aclose()
            except Exception:  # noqa: BLE001 —— 关闭失败不应影响退出流程
                logger.debug("关闭旧连接池失败", exc_info=True)
        self._retired.clear()

    # ------------------------------------------------------------------ 调用
    async def chat(
        self,
        messages: Sequence[ChatMessage | dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        task: str = "",
    ) -> LLMResponse:
        """发起一次 chat 补全。

        Args:
            messages:    对话消息（ChatMessage 或已组装好的 dict）。
            tools:       原生函数调用声明（OpenAI tools 格式）；给了模型可返回 tool_calls。
            json_mode:   置 True 时要求返回 JSON 对象（response_format）。
            temperature: 覆盖默认温度。
            max_tokens:  覆盖默认上限。
            task:        任务标签（monologue/decision/reflection/summary/...），
                         仅用于用量统计分组；FakeLLM 等替身可不实现该参数。
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_payload() if isinstance(m, ChatMessage) else m for m in messages],
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if json_mode:
            # 兼容协议：要求服务端返回严格 JSON 对象
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"
        client = self._ensure_client()

        try:
            # asyncio.timeout 是对老代码「无超时、可能永久挂起」的兜底修复
            async with asyncio.timeout(self.timeout):
                resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except asyncio.TimeoutError as e:
            # 整体超时（连接+发送+等响应）：思考模型生成慢时最常命中这里
            raise LLMError(
                f"LLM 调用超时（整体 >{self.timeout}s，多为模型生成过慢）：{url}") from e
        except httpx.HTTPStatusError as e:
            body = e.response.text or ""
            # 中转把「结构化输出被截断」当硬错误（如 502 structured_output_failed）时，
            # 归为可重试的 LLMTruncatedError，让上层自动加预算重试，而不是直接判失败。
            if _looks_truncated(body):
                raise LLMTruncatedError(
                    f"LLM 输出被截断（HTTP {e.response.status_code}）：{body[:200]}") from e
            raise LLMError(f"LLM 返回错误状态 {e.response.status_code}：{body[:200]}") from e
        except httpx.ConnectTimeout as e:
            # 连不上服务端（TCP 握手超时）：地址/端口/网络问题，与「生成慢」区分
            raise LLMError(f"LLM 连接超时（连不上服务端，检查地址与网络）：{url}") from e
        except httpx.ConnectError as e:
            # 连接被拒 / 域名不可达 / DNS 解析失败
            raise LLMError(f"LLM 无法连接服务端（连接被拒或地址不可达）：{url}") from e
        except httpx.ReadTimeout as e:
            # 已连上，但服务端迟迟不返回（生成阶段卡住）
            raise LLMError(f"LLM 读取超时（服务端已连接但迟迟不返回）：{url}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"LLM 网络错误：{e}") from e
        except json.JSONDecodeError as e:
            raise LLMError("LLM 响应不是合法 JSON") from e

        resp = self._parse_response(data)
        # 200 但被截断：JSON 模式下的输出必然不完整，抛可重试的 LLMTruncatedError，
        # 由 structured_chat 加预算重试（普通文本模式不抛，保留「截断但可用」的旧行为）。
        if json_mode and resp.finish_reason == "length":
            raise LLMTruncatedError(
                f"LLM 输出被截断（finish_reason=length，max_tokens 不足）：{resp.content[:120]}")
        await self._report_usage(resp, task)    # 用量回调（阶段二；未设置则无操作）
        return resp

    async def _report_usage(self, resp: LLMResponse, task: str) -> None:
        """把本次调用的用量交给上层（默认关闭 → 阶段一行为完全不变）。

        容错原则：用量统计属于「旁路观测」，失败绝不能影响主对话链路。
        回调既支持同步函数也支持协程函数，便于 AdminStore 用 to_thread 写库。
        """
        callback = self.on_usage
        if callback is None:
            return
        try:
            result = callback(self.provider_id, resp.model or self.model,
                              dict(resp.usage or {}), task)
            if asyncio.iscoroutine(result):     # 异步回调：等它写完
                await result
        except Exception:  # noqa: BLE001 —— 旁路失败只记日志
            logger.exception("用量回调执行失败（不影响本次调用）")

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """把服务端 JSON 解析成 LLMResponse。"""
        try:
            choice = data["choices"][0]
            message = choice.get("message", {})
        except (KeyError, IndexError) as e:
            raise LLMError(f"LLM 响应结构异常：{str(data)[:200]}") from e

        content = message.get("content") or ""
        tool_call = self._parse_tool_call(message.get("tool_calls"))

        return LLMResponse(
            content=content,
            tool_call=tool_call,
            model=data.get("model", self.model),
            usage=data.get("usage", {}) or {},
            raw=message,
            finish_reason=str(choice.get("finish_reason") or ""),   # 截断检测用
        )

    @staticmethod
    def _parse_tool_call(raw_calls: Any) -> ToolCall | None:
        """解析 OpenAI 格式的 tool_calls，取第一个。

        arguments 是 JSON 字符串；解析失败时给出空参并记日志，
        不抛异常——否则一次畸形调用会打断整轮对话。
        """
        if not raw_calls:
            return None
        first = raw_calls[0]
        fn = first.get("function", {}) or {}
        name = fn.get("name", "")
        raw_args = fn.get("arguments", "") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except (json.JSONDecodeError, TypeError):
            logger.warning("工具参数不是合法 JSON：%s", raw_args)
            args = {}
        if not name:
            return None
        return ToolCall(name=name, args=args)

    # ------------------------------------------------------------------ 便捷封装
    async def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        task: str = "",
    ) -> LLMResponse:
        """单轮 system+user 的便捷调用（摘要、画像等后台任务用）。"""
        return await self.chat(
            [ChatMessage(role="system", content=system),
             ChatMessage(role="user", content=user)],
            json_mode=json_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            task=task,
        )