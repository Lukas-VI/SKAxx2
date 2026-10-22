"""视觉解析：把一张图片交给多模态模型，产出「描述 + 检索关键词 + 情绪」。

为什么复用 LLMGateway.chat 而不新增 SDK：
    gateway.chat 的 messages 参数允许直接传 OpenAI 原生 dict，
    多模态 content 数组（text + image_url）可以原样透传，零新增依赖。
配置上允许单独指定视觉模型（media.vision_model / base_url / api_key）：
   主对话模型未必支持图片（如纯文本 qwen2.5:7b），视觉走另一个多模态端点更稳。
"""
from __future__ import annotations

import base64
import logging
from typing import Any

from ..llm.gateway import LLMError
from ..llm.structured import StructuredError, decode_json, strip_code_fence

logger = logging.getLogger(__name__)

VISION_SYSTEM = """你是一个表情包/图片标注器。请观察图片，输出一个 JSON 对象：
{
  "description": "一句话说明画面内容与它想表达的意思（中文，40 字内）",
  "tags": ["3~8 个中文检索关键词，覆盖情绪、动作、场景、梗"],
  "emotion": "最主要的情绪标签，如 开心/无语/疑惑/生气/安慰/得意",
  "is_sticker": true 或 false
}
只输出 JSON，不要解释。"""


def _plain_description(text: str, limit: int = 80) -> str:
    """模型没按 JSON 输出时，把纯文本压成一句可用的描述。

    多行/带代码围栏/多空白的输出直接塞进库里会很难读，这里先剥围栏再折叠空白，
    最后截断到 limit 字（截断点之后的内容对描述没有增量信息）。
    """
    line = " ".join(strip_code_fence(text).split())   # 先剥 ``` 围栏，再把所有空白压成单空格
    return line[:limit]                              # 截断，避免把整段解释都写进描述


class VisionDescriber:
    """图片 → 结构化语义（失败时返回空 dict，不打断主链）。"""

    def __init__(self, gateway: Any = None, cfg: dict[str, Any] | None = None) -> None:
        self.gateway = gateway
        cfg = cfg or {}
        # 没配置视觉网关时就关闭解析：宁可不解析，也不要拿纯文本模型硬塞图片报错。
        self.enabled = gateway is not None and bool(cfg.get("vision_enabled", True))
        # 默认给 1200：思考型模型会先耗掉一部分 token 在推理上，400 很容易被截成空回复
        self.max_tokens = int(cfg.get("vision_max_tokens", 1200))

    async def describe(self, data: bytes, mime: str = "image/jpeg") -> dict[str, Any]:
        """解析图片；任何失败都返回 {}（调用方据此保留「未解析」状态）。"""
        if not self.enabled or not data:
            return {}
        url = f"data:{mime or 'image/jpeg'};base64," + \
              base64.b64encode(data).decode("ascii")
        messages = [
            {"role": "system", "content": VISION_SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "请解析这张图片。"},
                {"type": "image_url", "image_url": {"url": url}},
            ]},
        ]
        try:
            resp = await self.gateway.chat(messages, task="vision",
                                           temperature=0.2,
                                           max_tokens=self.max_tokens)
        except LLMError as e:
            logger.info("视觉解析失败（保留原文未解析状态）：%s", e)
            return {}

        text = (resp.content or "").strip()
        if not text:
            # 空回复最常见的两个原因：① 思考型模型把 max_tokens 全用在推理上；
            # ② 中转丢弃了多模态 content。给出可操作的提示，别只留一句「失败」。
            logger.warning(
                "视觉模型返回空内容（finish_reason=%s, model=%s）："
                "可调大 media.vision_max_tokens（当前 %d）",
                resp.finish_reason, resp.model, self.max_tokens)
            return {}

        try:
            parsed = decode_json(text)
        except StructuredError:
            # 模型没按 JSON 输出（不少中转不支持 response_format）：退一步把纯文本
            # 直接当描述用。总比整条丢弃、最后只剩 QQ 的「[动画表情]」强。
            logger.info("视觉输出不是 JSON，改用纯文本作为描述：%s", text[:120])
            parsed = {"description": _plain_description(text), "tags": "",
                      "emotion": "", "is_sticker": False}
        if not isinstance(parsed, dict):
            return {}
        tags = parsed.get("tags") or []
        if isinstance(tags, list):
            tags_text = " ".join(str(t).strip() for t in tags if str(t).strip())
        else:
            tags_text = str(tags).strip()
        return {
            "description": str(parsed.get("description") or "").strip(),
            "tags": tags_text,
            "emotion": str(parsed.get("emotion") or "").strip(),
            "is_sticker": bool(parsed.get("is_sticker")),
        }
