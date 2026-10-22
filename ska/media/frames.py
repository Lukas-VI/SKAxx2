"""动图抽帧：把 GIF / 动态 WebP 转成一张静态 PNG，再交给视觉模型。

为什么必须抽帧：
    多模态接口基本只接受静态图；把动图直接丢过去，模型往往给不出画面内容，
    QQ 侧最终只留下 `[动画表情]` 这种没有信息量的摘要——等于没解析。

依赖取舍：
    Pillow 在项目的 venv 里本来就存在（不是本项目声明的依赖），这里做**可选导入**：
    装了就用它抽帧，没装就原样返回、由上层按原图尝试。这样既不新增
    requirements.txt 依赖，也不会因为环境差异让整条链路崩掉。
"""
from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

# 需要抽帧的格式：GIF 一定是动画；WebP 可能是动图，统一转静态最稳
ANIMATED_MIMES = frozenset({"image/gif", "image/webp"})

# 抽出来的静态图最长边：视觉模型按块计费，过大的图既慢又贵，768 足够看清表情
MAX_SIDE = 768


def still_frame(data: bytes, mime: str = "") -> tuple[bytes, str]:
    """把图片规范成「一张静态图」；返回 (字节, mime)。

    非动图、Pillow 不可用、解码失败时原样返回（调用方据此走原有逻辑）。
    """
    if not data or mime not in ANIMATED_MIMES:
        return data, mime or "image/jpeg"
    try:
        from PIL import Image          # 可选依赖：缺失时走下面的 ImportError 分支
    except ImportError:
        logger.debug("未安装 Pillow，动图将按原图交给视觉模型")
        return data, mime

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.seek(0)                       # 只取第一帧（表情包首帧通常就是主画面）
            frame = img.convert("RGBA")
            # 透明背景合成到白底：直接存 RGBA 某些接口会当透明区域是黑色，
            # 合成后画面更接近人眼看到的效果。
            canvas = Image.new("RGB", frame.size, (255, 255, 255))
            canvas.paste(frame, mask=frame.split()[-1])
            canvas.thumbnail((MAX_SIDE, MAX_SIDE))   # 等比缩放，不放大
            out = io.BytesIO()
            canvas.save(out, format="PNG")
            return out.getvalue(), "image/png"
    except Exception:  # noqa: BLE001 —— 解码失败就退回原图，不让抽帧拖垮解析
        logger.info("动图抽帧失败，改按原图解析", exc_info=True)
        return data, mime
