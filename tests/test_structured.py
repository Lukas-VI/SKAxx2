"""结构化输出「截断自动重试」测试（阶段三加固）。

背景（用户报告的缺陷）：LLM 返回 502，body 为
    {"error": {"type": "structured_output_failed",
               "message": "structured output was incomplete (finish_reason=length);
                           increase the output budget"}}
即模型输出被 max_tokens 截断，中转把不完整的 JSON 当硬错误抛出。

修复：把「截断」识别为**可重试的预算不足**（LLMTruncatedError），
由 structured_chat 自动按 2 倍抬高 max_tokens 再试，而不是直接降级沉默。

覆盖：
    * _looks_truncated：识别中转报错体里的截断特征；
    * LLMGateway：200+finish_reason=length（json 模式）、502 structured_output_failed
      都归为 LLMTruncatedError；非截断错误仍是普通 LLMError；
    * structured_chat：遇截断自动抬预算并恢复；预算达上限则放弃。
"""
from __future__ import annotations

import unittest

import httpx

from ska.llm.gateway import LLMError, LLMGateway, LLMTruncatedError, _looks_truncated
from ska.llm.models import ChatMessage, LLMResponse
from ska.llm.structured import MAX_STRUCTURED_TOKENS, structured_chat


# --------------------------------------------------------------------------- 载具
class _StaticClient:
    """替代 httpx.AsyncClient：post 一律返回预设的 Response。"""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.payload: dict | None = None

    async def post(self, url: str, json: dict | None = None) -> httpx.Response:
        self.payload = json
        return self._response


def _response(status: int, *, data: dict | None = None, text: str = "") -> httpx.Response:
    """造一个带 request 的 httpx.Response（raise_for_status 需要 request 才能抛）。"""
    req = httpx.Request("POST", "http://test/v1/chat/completions")
    if data is not None:
        return httpx.Response(status, json=data, request=req)
    return httpx.Response(status, text=text, request=req)


class _TruncatingLLM:
    """第一次调用抛「截断」，之后返回合法 JSON；记录每次的 max_tokens（鸭子类型替身）。"""

    def __init__(self, base: int = 1024) -> None:
        self.max_tokens = base                 # 模拟网关默认预算
        self.calls: list[dict] = []
        self.i = 0                             # 已发生的调用次数

    async def chat(self, messages, *, tools=None, json_mode=False,
                   temperature=None, max_tokens=None, task="") -> LLMResponse:
        self.calls.append({"max_tokens": max_tokens, "json_mode": json_mode, "task": task})
        if self.i == 0:                        # 首轮：模拟被截断
            self.i += 1
            raise LLMTruncatedError("输出被 max_tokens 截断（模拟）")
        self.i += 1
        return LLMResponse(content='{"action":"speak","content":"好"}', model="fake")


# --------------------------------------------------------------------------- 用例
class TestLooksTruncated(unittest.TestCase):
    """"报错体 → 是否截断" 的判定。"""

    def test_detects_relay_markers(self):
        body = ('{"error":{"type":"structured_output_failed",'
                '"message":"structured output was incomplete (finish_reason=length); '
                'increase the output budget"}}')
        self.assertTrue(_looks_truncated(body))
        self.assertTrue(_looks_truncated("done (finish_reason=length)"))
        self.assertTrue(_looks_truncated('{"finish_reason": "length"}'))

    def test_ignores_other_errors(self):
        self.assertFalse(_looks_truncated("invalid api key"))
        self.assertFalse(_looks_truncated(""))
        self.assertFalse(_looks_truncated("rate limit exceeded"))


class TestGatewayTruncationDetect(unittest.IsolatedAsyncioTestCase):
    """网关把「截断」与「普通错误」区分开。"""

    def _gw(self) -> LLMGateway:
        return LLMGateway({"max_tokens": 64})

    async def test_finish_reason_length_raises_truncated(self):
        data = {"choices": [{"message": {"content": '{"a":'}, "finish_reason": "length"}],
                "model": "m"}
        gw = self._gw()
        gw._client = _StaticClient(_response(200, data=data))
        with self.assertRaises(LLMTruncatedError):
            await gw.chat([ChatMessage(role="user", content="hi")], json_mode=True)

    async def test_length_without_json_mode_keeps_content(self):
        """非 json 模式：截断不抛，保留「截断但可用」的旧行为，同时暴露 finish_reason。"""
        data = {"choices": [{"message": {"content": "half"}, "finish_reason": "length"}],
                "model": "m"}
        gw = self._gw()
        gw._client = _StaticClient(_response(200, data=data))
        resp = await gw.chat([ChatMessage(role="user", content="hi")])
        self.assertEqual(resp.content, "half")
        self.assertEqual(resp.finish_reason, "length")

    async def test_relay_502_structured_output_failed(self):
        """中转把截断当硬错误返回 502 时，也归为可重试的 LLMTruncatedError。"""
        body = ('{"error":{"type":"structured_output_failed",'
                '"message":"structured output was incomplete (finish_reason=length); '
                'increase the output budget"}}')
        gw = self._gw()
        gw._client = _StaticClient(_response(502, text=body))
        with self.assertRaises(LLMTruncatedError):
            await gw.chat([ChatMessage(role="user", content="hi")], json_mode=True)

    async def test_other_http_error_is_plain_llm_error(self):
        gw = self._gw()
        gw._client = _StaticClient(_response(401, text="invalid api key"))
        with self.assertRaises(LLMError) as cm:
            await gw.chat([ChatMessage(role="user", content="hi")], json_mode=True)
        self.assertNotIsInstance(cm.exception, LLMTruncatedError)   # 不是截断类


class TestStructuredTruncationRetry(unittest.IsolatedAsyncioTestCase):
    """structured_chat：遇截断抬高预算重试。"""

    async def test_bumps_max_tokens_and_recovers(self):
        llm = _TruncatingLLM(base=1024)
        data = await structured_chat(llm, [ChatMessage(role="user", content="hi")], retry=1)
        self.assertEqual(data["action"], "speak")
        self.assertIsNone(llm.calls[0]["max_tokens"])        # 首轮不显式传预算
        self.assertEqual(llm.calls[1]["max_tokens"], 2048)   # 截断后按 2 倍抬升

    async def test_budget_capped_then_gives_up(self):
        """基准预算已达上限 → 无法再抬升，直接把截断错误抛给上层（由上层降级）。"""
        llm = _TruncatingLLM(base=MAX_STRUCTURED_TOKENS)
        with self.assertRaises(LLMTruncatedError):
            await structured_chat(llm, [ChatMessage(role="user", content="hi")], retry=3)
        self.assertEqual(len(llm.calls), 1)                  # 未做无意义的重复调用


if __name__ == "__main__":
    unittest.main()
