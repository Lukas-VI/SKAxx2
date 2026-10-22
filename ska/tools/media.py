"""图片相关工具：image_send（把一张图附到回复里）/ giphy_search（联网找 GIF）。

为什么 image_send 只返回一个 `[[IMAGE:path]]` 标记，而**不直接发消息**：
    工具不该绕过 Mind 的决策直接往渠道塞消息（否则会出现「工具发了图、Mind 还在
    组织文字」的双发/抢占）。这里的约定是：
        image_send 校验并登记图片 → AgentLoop 收集标记 → Mind 并入 CoreOutput.images
        → 渠道（QQ）统一组装 image 段。
    这样「画图 → 发图」是一条可控的链路，且工具失败时不会污染对话。

为什么本地路径必须过白名单：
    模型能写代码生成图片，就能被诱导去读本机任意文件（如密钥截图）。白名单在
    `media.allow_dirs`（+ 表情包目录），由 media/images.py 统一执行校验。
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from ..media.images import inspect_source, mark_image, send_dirs
from .base import Tool, tool

logger = logging.getLogger(__name__)

# Giphy 公共检索接口（官方文档：https://developers.giphy.com/docs/api/endpoint）
GIPHY_SEARCH_URL = "https://api.giphy.com/v1/gifs/search"


def _media_cfg(deps: Any) -> dict[str, Any]:
    """读取顶层 media 小节（取不到就返回空 dict，退回各字段的默认值）。"""
    cfg = getattr(deps, "cfg", None)
    try:
        return cfg.section("media") if cfg is not None else {}
    except Exception:  # noqa: BLE001 —— 配置结构异常不该让工具直接崩
        logger.debug("读取 media 配置失败", exc_info=True)
        return {}


# ===========================================================================
# image_send —— 把图片附加到本轮回复
# ===========================================================================
@tool("image_send")
class ImageSendTool(Tool):
    description = (
        "把一张图片附在本轮回复里一起发出（支持本地路径或 http(s) 链接）。"
        "上传后**不需要**再把路径写进回复文字，系统会自动随消息发送。"
        "本地路径必须在配置允许的目录内；常用场景：先用代码画出 PNG，再调用本工具。")

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "图片来源：本地绝对/相对路径（.png/.jpg/.gif/.webp），或 http(s) 链接",
            },
        },
        "required": ["source"],
    }

    async def run(self, source: str = "", **_: Any) -> str:
        """校验来源并返回图片标记；不合法时返回可读错误（让模型自己纠正）。"""
        media = _media_cfg(self.deps)
        max_bytes = int(media.get("max_image_bytes", 5 * 1024 * 1024))
        kind, value, error = inspect_source(source, send_dirs(media), max_bytes)
        if not kind:
            # 把「为什么失败」原样交给模型：它下一轮就能换路径/改后缀/先压缩图片。
            # 笼统的「来源不合法」只会让模型反复重试同一个错误路径。
            return f"图片发送失败：{error}"
        return f"{mark_image(value)}图片已附加，将在本轮回复中一起发出。"


# ===========================================================================
# giphy_search —— 联网找 GIF（本地表情库没有合适的时候用）
# ===========================================================================
@tool("giphy_search")
class GiphySearchTool(Tool):
    description = ("在 Giphy 上检索 GIF 动图，返回几个候选的直链与简介。"
                   "当你想发一个本地表情包库里没有的动图时使用；"
                   "拿到直链后再调用 image_send 把图片发出去。")

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "英文关键词效果更好，如 happy / speechless / cat"},
            "limit": {"type": "integer", "description": "返回条数，默认 3，最多 10"},
        },
        "required": ["query"],
    }

    timeout = 15.0

    async def run(self, query: str = "", limit: int = 3, **_: Any) -> str:
        text = str(query or "").strip()
        if not text:
            return "检索词为空。"
        api_key = str(_media_cfg(self.deps).get("giphy_api_key") or "").strip()
        if not api_key:
            return ("没有配置 media.giphy_api_key，无法联网检索 GIF。"
                    "可以先用本地表情包或直接说话。")
        try:
            count = min(max(int(limit or 3), 1), 10)
        except (TypeError, ValueError):
            count = 3
        params = {"api_key": api_key, "q": text, "limit": count, "rating": "pg-13",
                  "lang": "en"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
                resp = await client.get(GIPHY_SEARCH_URL, params=params)
                resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            logger.info("Giphy 检索失败：%s", e)
            return f"Giphy 检索失败：{e}"

        items = (data.get("data") or []) if isinstance(data, dict) else []
        if not items:
            return f"Giphy 上没有找到「{text}」相关的动图。"
        lines: list[str] = []
        for i, item in enumerate(items, 1):
            images = (item.get("images") or {}) if isinstance(item, dict) else {}
            # 优先 fixed_height（体积小、加载快）；缺失再退回原图
            url = str((images.get("fixed_height") or {}).get("url")
                      or (images.get("original") or {}).get("url") or "")
            title = str(item.get("title") or "").strip() or "（无标题）"
            if url:
                lines.append(f"{i}. {title} — {url}")
        return "\n".join(lines) if lines else f"Giphy 返回了结果，但没有可用的图片直链（{text}）。"
