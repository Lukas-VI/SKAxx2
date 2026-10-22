"""阶段四：图片 IO 与本地表情包库。

三个小模块各管一件事：
    images   —— 图片路径/URL 校验、base64 编码、OneBot image 段构造、工具标记解析
    vision   —— 用视觉模型把图片解析成「描述 + 关键词 + 情绪」
    stickers —— 表情包入库（下载、sha256 去重、首次视觉解析）与语义检索
"""
from .images import (IMAGE_MARKER_RE, build_message, extract_image_markers,
                     inspect_source, mark_image, resolve_source, send_dirs)
from .stickers import StickerBook
from .vision import VisionDescriber

__all__ = [
    "IMAGE_MARKER_RE", "build_message", "extract_image_markers", "inspect_source",
    "mark_image", "resolve_source", "send_dirs", "StickerBook", "VisionDescriber",
]
