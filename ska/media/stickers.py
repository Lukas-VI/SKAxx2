"""本地表情包库：下载 → sha256 去重 → 首次视觉解析 → 入库 / 检索。

核心取舍：
    * 去重键用图片 **内容 sha256**，不用 URL——QQ 的图片 URL 带时效签名，
      同一个人重发同一个表情，URL 往往不同，只有内容哈希才稳定；
    * 只有「表情包」类图片入库（OneBot sub_type=1 / mface / face），
      普通照片只做一次性视觉描述，不进表情包库，避免把私照当表情发出去；
    * 视觉解析很慢（一次多模态调用），所以命中缓存后绝不再跑第二次。
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from .frames import still_frame
from .vision import VisionDescriber

logger = logging.getLogger(__name__)

# 注入用的下载函数签名（测试里可换成假实现，避免真的联网）
Fetcher = Callable[[str, int], Awaitable[bytes | None]]

# 未解析时的占位描述：仍然入库（保留图片），等以后有视觉模型再补
UNPARSED = "（未解析的表情包）"


def _sniff_ext(data: bytes) -> str:
    """按文件头猜图片后缀（QQ 图片 URL 常常没有扩展名）。"""
    if data.startswith(b"\x89PNG"):
        return ".png"
    if data.startswith(b"GIF8"):
        return ".gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def _sniff_mime(data: bytes) -> str:
    """按文件头判断真实 MIME。

    不能一律写死 image/jpeg：data URL 里声明成 jpeg、实际是 GIF，
    严格的视觉接口会直接拒收，宽松的也会被当成损坏图片处理。
    """
    if data.startswith(b"\x89PNG"):
        return "image/png"
    if data.startswith(b"GIF8"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"BM"):
        return "image/bmp"
    return "image/jpeg"


async def _http_fetch(url: str, max_bytes: int) -> bytes | None:
    """流式下载，超过 max_bytes 立即放弃（防止被超大图拖死）。"""
    if not url.lower().startswith(("http://", "https://")):
        return None
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code != 200:
                    logger.info("下载表情包失败：HTTP %d %s", resp.status_code, url)
                    return None
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        logger.info("表情包超过大小上限，放弃：%s", url)
                        return None
                    chunks.append(chunk)
                return b"".join(chunks)
    except httpx.HTTPError as e:
        logger.info("下载表情包网络错误：%s（%s）", e, url)
        return None


class StickerBook:
    """表情包库门面：Mind 用它解析来图 / 挑选表情，测试可注入假下载。"""

    def __init__(self, store: Any, cfg: dict[str, Any] | None = None,
                 vision: VisionDescriber | None = None, *,
                 fetch: Fetcher | None = None) -> None:
        self.store = store
        cfg = cfg or {}
        self.cfg = cfg
        self.max_bytes = int(cfg.get("max_image_bytes", 5 * 1024 * 1024))
        self.vision = vision
        self.fetch: Fetcher = fetch or _http_fetch
        self.dir = self._resolve_dir(cfg.get("sticker_dir", "data/stickers"))

    @staticmethod
    def _resolve_dir(raw: str) -> Path:
        """表情包保存目录：相对路径按项目根解析。"""
        from ..kernel.config import package_root
        path = Path(str(raw or "data/stickers"))
        if not path.is_absolute():
            path = package_root() / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------ 来图解析
    async def analyze(self, image: dict[str, Any], *, user: str = "",
                      channel: str = "") -> str:
        """解析一条来图，返回给 LLM 看的文字描述（失败返回空串）。

        image 由 QQ 渠道解析 OneBot 消息段得到，至少含 kind/url/summary 之一。
        """
        if not isinstance(image, dict):
            return ""
        kind = str(image.get("kind") or "image")
        summary = str(image.get("summary") or "").strip()
        url = str(image.get("url") or "").strip()

        # QQ 内置表情（face/mface）通常没有图片 URL，只能存元数据 + 官方摘要
        if kind in ("face", "mface") and not url:
            face_id = str(image.get("id") or image.get("face_id") or summary or "unknown")
            key = f"face:{face_id}"
            row = await self.store.get_sticker(key)
            if row:
                return str(row.get("description") or UNPARSED)
            description = summary or f"QQ 内置表情（face={face_id}）"
            await self.store.add_sticker(
                key, kind=kind, description=description, tags="表情 QQ内置表情",
                user=user, channel=channel)
            return description

        if not url:
            return ""
        data = await self.fetch(url, self.max_bytes)
        if not data:
            return ""
        digest = hashlib.sha256(data).hexdigest()

        row = await self.store.get_sticker(digest)
        if row:
            # 命中缓存：绝不重复跑视觉模型（同一个表情发的越多越省）
            await self.store.touch_sticker(int(row["id"]))
            cached = str(row.get("description") or "")
            if cached and cached != UNPARSED:
                return cached
            # 历史坏数据（当初视觉失败/模型不可用留下的占位）：现在有能力就补一次，
            # 否则这个表情永远只能回「（未解析的表情包）」，这也是「失配」的来源之一。
            if self.vision is None:
                return cached or UNPARSED
            frame, frame_mime = still_frame(data, _sniff_mime(data))
            parsed = await self.vision.describe(frame, mime=frame_mime)
            fixed = str(parsed.get("description") or "")
            if fixed:
                await self.store.update_sticker(
                    int(row["id"]), description=fixed,
                    tags=str(parsed.get("tags") or ""),
                    emotion=str(parsed.get("emotion") or ""))
                return fixed
            return cached or UNPARSED

        # 动图先抽一帧静态图：视觉接口读不了动画，直接送 GIF 只会得到
        # 「[动画表情]」这种无信息结果。抽帧失败时 still_frame 会原样返回。
        frame, frame_mime = still_frame(data, _sniff_mime(data))
        parsed = await self.vision.describe(frame, mime=frame_mime) if self.vision else {}
        description = str(parsed.get("description") or summary or UNPARSED)
        tags = str(parsed.get("tags") or "").strip()
        emotion = str(parsed.get("emotion") or "").strip()

        # 普通照片：只给描述，不进表情包库（防止把私照当表情包发出去）。
        # face/mface 若带图 URL，同样算表情包（QQ 内置表情也有可下载的图）。
        if kind not in ("sticker", "face", "mface"):
            return description

        ext = _sniff_ext(data)
        path = self.dir / f"{digest}{ext}"
        saved = ""
        try:
            if not path.exists():
                path.write_bytes(data)
            saved = str(path)                 # 只有写成功才记路径（空串=发不出去）
        except OSError:
            logger.warning("保存表情包失败：%s", path, exc_info=True)
        await self.store.add_sticker(
            digest, file=saved, url=url, kind="sticker",
            description=description, tags=tags, emotion=emotion,
            user=user, channel=channel)
        return description

    # ------------------------------------------------------------------ 选择表情
    async def pick(self, query: str, *, user: str = "") -> dict[str, Any] | None:
        """按语义挑一个表情包；query 为空时退化到「最近常用」。"""
        text = str(query or "").strip()
        if text:
            rows = await self.store.search_stickers(text, top_k=1, user=user)
        else:
            rows = await self.store.list_stickers(limit=1)
        if not rows:
            return None
        row = rows[0]
        # 没有本地文件的内置表情发不出去（无法构造 image 段），直接跳过
        if not str(row.get("file") or "").strip():
            return None
        await self.store.touch_sticker(int(row["id"]))
        return row

    async def list(self, limit: int = 100) -> list[dict[str, Any]]:
        return await self.store.list_stickers(limit=limit)

    # ------------------------------------------------------------------ 后台管理
    async def import_one(self, data: bytes, *, name: str = "", user: str = "",
                         channel: str = "webui") -> dict[str, Any]:
        """导入一张本地表情包（后台「批量导入」用）。

        与 analyze() 的区别：来源是上传的字节，不需要下载；其余流程完全一致
        （sha256 去重 → 动图抽帧 → 视觉嵌入 → 落盘入库）。
        返回 {"status": added|skipped|error, ...}，批量导入据此统计成功/跳过。
        """
        if not data:
            return {"status": "error", "reason": "空文件"}
        digest = hashlib.sha256(data).hexdigest()
        row = await self.store.get_sticker(digest)
        if row:
            await self.store.touch_sticker(int(row["id"]))
            return {"status": "skipped", "id": int(row["id"]),
                    "description": str(row.get("description") or "")}
        try:
            frame, frame_mime = still_frame(data, _sniff_mime(data))
            parsed = await self.vision.describe(frame, mime=frame_mime) if self.vision else {}
        except Exception:  # noqa: BLE001 —— 单张失败不拖垮整批
            logger.exception("导入表情包时视觉解析失败：%s", name)
            parsed = {}
        description = str(parsed.get("description") or Path(name).stem or UNPARSED)
        ext = _sniff_ext(data)
        path = self.dir / f"{digest}{ext}"
        saved = ""
        try:
            if not path.exists():
                path.write_bytes(data)
            saved = str(path)
        except OSError:
            logger.warning("保存导入的表情包失败：%s", path, exc_info=True)
        sid = await self.store.add_sticker(
            digest, file=saved, kind="sticker", description=description,
            tags=str(parsed.get("tags") or ""), emotion=str(parsed.get("emotion") or ""),
            user=user, channel=channel)
        return {"status": "added", "id": sid, "description": description}

    async def update(self, sticker_id: int, *, description: str | None = None,
                     tags: str | None = None, emotion: str | None = None) -> bool:
        """手动改「嵌入信息」（描述/标签/情绪）——失配时最有效的修补手段。"""
        return await self.store.update_sticker(sticker_id, description=description,
                                               tags=tags, emotion=emotion)

    async def delete(self, sticker_id: int) -> bool:
        """删除表情包记录，并尽量删掉本地文件（只删库目录内的，防误删）。"""
        row = await self.store.get_sticker_by_id(sticker_id)
        if not row:
            return False
        ok = await self.store.delete_sticker(sticker_id)
        path = Path(str(row.get("file") or ""))
        if ok and path.is_file():
            try:
                if path.resolve().is_relative_to(self.dir.resolve()):
                    path.unlink()
            except (OSError, ValueError):
                logger.warning("删除表情包文件失败：%s", path, exc_info=True)
        return ok

    async def image(self, sticker_id: int) -> tuple[bytes, str] | None:
        """读取本地图片字节（后台列表缩略图用）；文件不存在或不在库目录内返回 None。"""
        row = await self.store.get_sticker_by_id(sticker_id)
        path = Path(str((row or {}).get("file") or ""))
        if not path.is_file():
            return None
        try:
            if not path.resolve().is_relative_to(self.dir.resolve()):
                return None
        except ValueError:
            return None
        data = path.read_bytes()               # 只读一次，MIME 直接从字节头判断
        return data, _sniff_mime(data)
