"""违禁词过滤插件（迁移自老 core/message_filter.py）。

变化点：
    老实现：在业务代码里手动调用 filter_message()。
    新实现：订阅总线的 output 事件——任何要发出去的消息都会先经过这里，
            「发送前净化」变成不可绕过的横切能力。

本插件只做替换（违禁词 → *），与老实现行为一致：不拦截、不改路由。
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from ..kernel.bus import EventBus
from ..kernel.config import package_root
from ..kernel.event import TOPIC_OUTPUT

logger = logging.getLogger(__name__)

# 违禁词表位置（相对项目根）
BANNED_WORDS_FILE = "config/banned_words.json"


class _Filter:
    """违禁词与敏感模式的替换器。"""

    def __init__(self, cfg) -> None:
        self._words: set[str] = set()
        self._patterns: list[re.Pattern] = []
        self._load(Path(package_root()) / BANNED_WORDS_FILE)

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("读取违禁词表失败，过滤将不生效：%s", e)
            return
        self._words = set(data.get("words", []))
        self._patterns = [re.compile(p, re.IGNORECASE) for p in data.get("patterns", [])]

    def sanitize(self, text: str) -> str:
        """把违禁词与敏感模式替换成等长星号。"""
        for word in self._words:
            if word in text:
                text = text.replace(word, "*" * len(word))
        for pattern in self._patterns:
            text = pattern.sub(lambda m: "*" * len(m.group()), text)
        return text

    def on_output(self, event) -> None:
        """output 事件处理器：原地清洗 CoreOutput.content。"""
        out = event.payload
        content = getattr(out, "content", None)
        if isinstance(content, str):
            cleaned = self.sanitize(content)
            if cleaned != content:
                logger.info("已过滤输出中的敏感内容")
                out.content = cleaned


def install(bus: EventBus, cfg) -> _Filter:
    """把过滤器挂到 output 事件上。"""
    filt = _Filter(cfg)
    bus.subscribe(TOPIC_OUTPUT, filt.on_output)
    logger.info("filter 插件已装载")
    return filt