"""图片 IO 工具：来源校验、OneBot 消息段构造、工具标记解析。

为什么要有「来源校验」：
    模型可以让我们把任意本地文件发到 QQ，这等于给它一个「读取本机文件」的后门。
    所以本地路径必须落在 media.allow_dirs 白名单内；http(s) / data: URL 直接放行。

为什么本地文件转成 base64://：
    OneBot v11 的 image 段约定支持 http(s)://、base64://、file:// 与本地路径，
    但不同实现的「本地路径」解释不一致（尤其 Windows 盘符）。base64 最稳，
    且 SnowLuma 与 SKAxx2 跑在同一台机器，不存在跨机传输成本。
"""
from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# 工具/决策里表示「要附加一张图」的标记：
#   image_send 工具返回它，AgentLoop 收集后从文本里剥离，Mind 再并入 CoreOutput.images。
# 用显式标记而不是让模型复述路径，避免模型改写/截断导致发错文件。
IMAGE_MARKER_RE = re.compile(r"\[\[IMAGE:(.+?)\]\]", re.S)

# 可发送的图片后缀（白名单，避免把可执行文件当图片读出来）
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# 后缀 → MIME（视觉模型与 base64 段需要）
_MIME = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}


def mark_image(source: str) -> str:
    """生成图片附加标记（供工具返回给 AgentLoop 收集）。"""
    return f"[[IMAGE:{source}]]"


def extract_image_markers(text: str) -> tuple[str, list[str]]:
    """从一段文本里抽出所有图片标记，返回 (去掉标记的文本, 来源列表)。

    AgentLoop 用它把「工具想发的图」提升为结构化字段，同时不把内部标记
    泄漏到用户可见的回复里。
    """
    sources = [m.group(1).strip() for m in IMAGE_MARKER_RE.finditer(text or "")]
    cleaned = IMAGE_MARKER_RE.sub("", text or "").strip()
    return cleaned, [s for s in sources if s]


def is_url(source: str) -> bool:
    """是否为可直接交给 OneBot 的远程资源。"""
    low = (source or "").strip().lower()
    return low.startswith(("http://", "https://", "data:", "base64://"))


def _allowed_roots(allow_dirs: Iterable[str]) -> list[Path]:
    """把配置里的白名单目录解析成绝对路径（相对路径按项目根解析）。"""
    from ..kernel.config import package_root
    roots: list[Path] = []
    for raw in allow_dirs or []:
        text = str(raw or "").strip()
        if not text:
            continue
        path = Path(os.path.expandvars(os.path.expanduser(text)))
        if not path.is_absolute():
            path = package_root() / path
        try:
            roots.append(path.resolve())
        except OSError:
            continue
    return roots


def _mb(size: int) -> str:
    """把字节数格式化成 MB，用于错误提示（让模型/人一眼看出差多少）。"""
    return f"{size / 1048576:.1f} MB"


def inspect_source(source: str, allow_dirs: Iterable[str] = (),
                   max_bytes: int = 5 * 1024 * 1024
                   ) -> tuple[str, str, str]:
    """校验图片来源，并给出**具体**的失败原因（而不是笼统的「来源不合法」）。

    Returns:
        (kind, value, error)
            kind  = "url" / "file" / ""（空串表示不合法）
            value = 原地址或解析后的绝对路径
            error = 中文原因（合法时为空串）

    为什么要把原因拆开：模型拿到「来源不合法」根本无从纠正；用户在 MindLine
    里也分不清是路径不存在、后缀不对、超过大小，还是白名单没生效。分开之后，
    工具可以直接把「路径不存在 / 超过 5MB / 白名单里没有这个目录」回给模型。
    """
    text = str(source or "").strip().strip('"').strip("'")
    if not text:
        return ("", "", "没有提供图片来源（source 为空）")
    if is_url(text):
        return ("url", text, "")

    path = Path(os.path.expandvars(os.path.expanduser(text)))
    if not path.is_absolute():
        from ..kernel.config import package_root
        path = package_root() / path
    try:
        path = path.resolve()
    except OSError as e:                       # 盘符不存在、路径过长等
        return ("", "", f"路径无法解析（{e}）：{text}")

    if not path.exists():
        return ("", "", f"文件不存在：{path}。请确认路径拼写和盘符；"
                        "也可以改用 http(s) 链接。")
    if not path.is_file():
        return ("", "", f"这不是一个文件（可能是目录）：{path}")
    suffix = path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        allowed = "、".join(sorted(IMAGE_SUFFIXES))
        return ("", "", f"不支持的图片后缀「{suffix or '（无后缀）'}」：{path}；"
                        f"只接受 {allowed}")
    try:
        size = path.stat().st_size
    except OSError as e:
        return ("", "", f"读取文件信息失败：{path}（{e}）")
    limit = max(int(max_bytes), 1)
    if size > limit:
        return ("", "", f"图片太大：{_mb(size)}，超过上限 {_mb(limit)}"
                        "（media.max_image_bytes，可在 config.yaml 调整）")

    roots = _allowed_roots(allow_dirs)
    # 白名单为空 = 一律拒绝（与 code.allow_dirs 的保守策略一致）
    if not roots:
        return ("", "", "media.allow_dirs 白名单为空：本地图片一律拒绝"
                        "（只放行 http(s) 链接与表情包目录）。请把图片所在目录写进"
                        " config/config.yaml 的 media.allow_dirs，并重启进程。")
    if not any(path.is_relative_to(root) for root in roots):
        dirs = "、".join(str(r) for r in roots)
        return ("", "", f"路径不在 media.allow_dirs 白名单内：{path}；"
                        f"当前生效的白名单：{dirs}。若刚改过 config/config.yaml，"
                        "需要重启进程才会生效。")
    return ("file", str(path), "")


def resolve_source(source: str, allow_dirs: Iterable[str] = (),
                   max_bytes: int = 5 * 1024 * 1024
                   ) -> tuple[str, str] | None:
    """校验一个图片来源（兼容旧调用：只关心是否合法）。

    Returns:
        ("url", 原地址) 或 ("file", 绝对路径)；不合法返回 None。
    """
    kind, value, error = inspect_source(source, allow_dirs, max_bytes)
    if not kind:
        if error:
            logger.warning("图片来源被拒绝：%s", error)
        return None
    return (kind, value)


def mime_of(path: str | Path) -> str:
    """按后缀猜 MIME；未知按 jpeg 兜底。"""
    return _MIME.get(Path(path).suffix.lower(), "image/jpeg")


def to_base64_source(path: str | Path) -> str:
    """把本地图片读成 OneBot 可用的 base64:// 段。"""
    with open(path, "rb") as fh:
        return "base64://" + base64.b64encode(fh.read()).decode("ascii")


def image_segment(source: str, *, max_bytes: int = 5 * 1024 * 1024,
                  allow_dirs: Iterable[str] = ()) -> dict[str, Any] | None:
    """把一个来源转成 OneBot image 段；不合法返回 None（并记下具体原因）。"""
    kind, value, error = inspect_source(source, allow_dirs, max_bytes)
    if not kind:
        # 渠道层是最后一道闸：这里静默丢弃会让人以为「什么都没发生」，必须留痕
        logger.warning("图片被跳过：%s", error or source)
        return None
    if kind == "url":
        return {"type": "image", "data": {"file": value}}
    try:
        return {"type": "image", "data": {"file": to_base64_source(value)}}
    except OSError:
        logger.warning("读取图片失败：%s", value, exc_info=True)
        return None


def send_dirs(media_cfg: dict[str, Any]) -> list[str]:
    """计算「允许发送」的本地目录：media.allow_dirs + 表情包目录。

    表情包目录必须始终在白名单里：它是程序自己下载保存的（内容已校验），
    若还要用户手动把它写进 allow_dirs，会变成一个很容易踩的配置坑
    （表现为「表情包入库成功但永远发不出去」）。
    """
    dirs = [str(x).strip() for x in (media_cfg.get("allow_dirs") or []) if str(x or "").strip()]
    sticker_dir = str(media_cfg.get("sticker_dir") or "").strip()
    if sticker_dir:
        dirs.append(sticker_dir)
    return dirs


def build_message(content: str, images: Iterable[str] = (), *,
                  max_bytes: int = 5 * 1024 * 1024,
                  allow_dirs: Iterable[str] = (),
                  max_images: int = 4) -> list[dict[str, Any]]:
    """组装 OneBot 消息段数组：文本（可空）+ 最多 max_images 张图。"""
    segments: list[dict[str, Any]] = []
    text = str(content or "").strip()
    if text:
        segments.append({"type": "text", "data": {"text": text}})
    for source in list(images or [])[:max(int(max_images), 0)]:
        seg = image_segment(source, max_bytes=max_bytes, allow_dirs=allow_dirs)
        if seg is not None:
            segments.append(seg)
    return segments
