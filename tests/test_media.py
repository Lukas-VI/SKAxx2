"""阶段四：图片 IO 与本地表情包库的单元测试。

覆盖四条链路的关键不变量：
    1. 图片标记的生成/剥离（工具 → AgentLoop → Mind 的交接协议）；
    2. 图片来源白名单（本地路径越界必须拒绝，URL 放行）；
    3. OneBot 消息段组装（文本 + base64:// 图片）；
    4. 表情包去重（同内容只跑一次视觉）与检索。
"""
from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path

from ska.io.qq import QQChannel
from ska.kernel.context import SessionContext
from ska.kernel.packet import CoreInput
from ska.llm.models import LLMResponse
from ska.loop.agent import AgentLoop
from ska.loop.mind import Mind
from ska.media.frames import still_frame
from ska.media.images import (build_message, extract_image_markers, inspect_source,
                              mark_image, resolve_source, send_dirs)
from ska.media.stickers import StickerBook
from ska.media.vision import VisionDescriber
from ska.memory.store import MemoryStore
from ska.tools.base import ToolDeps
from ska.tools.media import ImageSendTool

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"0" * 64      # 只要文件头对，后缀嗅探即可识别


def _gif_bytes() -> bytes:
    """用 Pillow 现场生成一个 2 帧 GIF（测试数据不落盘成二进制文件）。"""
    from PIL import Image                           # 未安装时抛 ImportError，由用例跳过

    frames = [Image.new("RGB", (4, 4), (255, 0, 0)),
              Image.new("RGB", (4, 4), (0, 255, 0))]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True,
                   append_images=frames[1:], duration=100, loop=0)
    return buf.getvalue()


class FakeVision:
    """假视觉模型：记录调用次数，返回固定解析结果。"""

    def __init__(self) -> None:
        self.calls = 0

    async def describe(self, data: bytes, mime: str = "image/jpeg") -> dict:
        self.calls += 1
        return {"description": "开心地笑", "tags": "开心 笑 表情包",
                "emotion": "开心", "is_sticker": True}


class TestImageMarkers(unittest.TestCase):
    def test_marker_roundtrip(self):
        text = f"已画好{mark_image('E:/SKA-sandbox/a.png')}，请查收"
        cleaned, sources = extract_image_markers(text)
        self.assertEqual(sources, ["E:/SKA-sandbox/a.png"])
        self.assertNotIn("[[IMAGE:", cleaned)          # 内部标记不得泄漏到用户可见文本
        self.assertIn("已画好", cleaned)

    def test_no_marker(self):
        cleaned, sources = extract_image_markers("普通文本")
        self.assertEqual(cleaned, "普通文本")
        self.assertEqual(sources, [])


class TestResolveSource(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.abspath(self._tmp.name)
        self.file = os.path.join(self.dir, "a.png")
        with open(self.file, "wb") as fh:
            fh.write(PNG_BYTES)

    def tearDown(self):
        self._tmp.cleanup()

    def test_url_passes(self):
        self.assertEqual(resolve_source("https://x/a.gif"),
                         ("url", "https://x/a.gif"))

    def test_local_file_inside_allow_dirs(self):
        self.assertEqual(resolve_source(self.file, [self.dir]), ("file", self.file))

    def test_local_file_outside_allow_dirs_rejected(self):
        # 白名单为空 = 一律拒绝（防止模型借发图读取本机任意文件）
        self.assertIsNone(resolve_source(self.file, []))

    def test_non_image_suffix_rejected(self):
        txt = os.path.join(self.dir, "secret.txt")
        with open(txt, "w", encoding="utf-8") as fh:
            fh.write("x")
        self.assertIsNone(resolve_source(txt, [self.dir]))

    def test_send_dirs_includes_sticker_dir(self):
        dirs = send_dirs({"allow_dirs": ["E:/x"], "sticker_dir": "data/stickers"})
        self.assertIn("E:/x", dirs)
        self.assertIn("data/stickers", dirs)


class TestInspectSource(unittest.TestCase):
    """失败原因必须可区分：路径 / 后缀 / 大小 / 白名单，不能只说「不合法」。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.abspath(self._tmp.name)
        self.file = os.path.join(self.dir, "a.png")
        with open(self.file, "wb") as fh:
            fh.write(PNG_BYTES)

    def tearDown(self):
        self._tmp.cleanup()

    def test_ok_file_has_no_error(self):
        kind, value, error = inspect_source(self.file, [self.dir])
        self.assertEqual(kind, "file")
        self.assertEqual(error, "")
        self.assertTrue(os.path.isabs(value))

    def test_missing_file(self):
        path = os.path.join(self.dir, "nope.jpg")
        kind, _, error = inspect_source(path, [self.dir])
        self.assertEqual(kind, "")
        self.assertIn("文件不存在", error)
        self.assertIn("nope.jpg", error)                 # 回显解析后的绝对路径

    def test_directory_is_not_a_file(self):
        kind, _, error = inspect_source(self.dir, [self.dir])
        self.assertEqual(kind, "")
        self.assertIn("不是一个文件", error)

    def test_unsupported_suffix(self):
        txt = os.path.join(self.dir, "secret.txt")
        with open(txt, "w", encoding="utf-8") as fh:
            fh.write("x")
        kind, _, error = inspect_source(txt, [self.dir])
        self.assertEqual(kind, "")
        self.assertIn(".txt", error)
        self.assertIn(".png", error)                     # 告诉模型能用什么

    def test_too_large_reports_sizes(self):
        kind, _, error = inspect_source(self.file, [self.dir], max_bytes=8)
        self.assertEqual(kind, "")
        self.assertIn("图片太大", error)
        self.assertIn("超过上限", error)

    def test_empty_whitelist(self):
        kind, _, error = inspect_source(self.file, [])
        self.assertEqual(kind, "")
        self.assertIn("白名单为空", error)

    def test_outside_whitelist_lists_effective_dirs(self):
        other = os.path.join(self._tmp.name, "other")
        os.makedirs(other)
        kind, _, error = inspect_source(self.file, [other])
        self.assertEqual(kind, "")
        self.assertIn("不在 media.allow_dirs 白名单内", error)
        self.assertIn("当前生效的白名单", error)
        self.assertIn(str(Path(other).resolve()), error)
        self.assertIn("重启", error)                      # 提示改完配置要重启

    def test_empty_source(self):
        _, _, error = inspect_source("")
        self.assertIn("没有提供图片来源", error)


class _MediaCfg:
    """image_send 工具用的最小配置替身。"""

    def __init__(self, media: dict) -> None:
        self._media = media

    def section(self, name: str) -> dict:
        return self._media if name == "media" else {}


class TestImageSendFeedback(unittest.IsolatedAsyncioTestCase):
    """image_send 的失败反馈要具体到「哪一种失败」，模型才知道怎么改。"""

    async def _run(self, source: str, media: dict) -> str:
        tool = ImageSendTool(ToolDeps(cfg=_MediaCfg(media)))
        return await tool.run(source=source)

    async def test_outside_whitelist_explains(self):
        msg = await self._run("C:/definitely/not/here.png", {"allow_dirs": ["E:/x"]})
        self.assertIn("文件不存在", msg)                  # 先报「不存在」而不是白名单
        self.assertNotIn("来源不合法", msg)

    async def test_success_returns_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.png")
            with open(path, "wb") as fh:
                fh.write(PNG_BYTES)
            msg = await self._run(path, {"allow_dirs": [tmp]})
        self.assertIn(f"[[IMAGE:{Path(path).resolve()}]]", msg)


class TestBuildMessage(unittest.TestCase):
    def test_text_and_base64_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.png")
            with open(path, "wb") as fh:
                fh.write(PNG_BYTES)
            segs = build_message("看图", [path], allow_dirs=[tmp])
        self.assertEqual(segs[0], {"type": "text", "data": {"text": "看图"}})
        self.assertEqual(segs[1]["type"], "image")
        self.assertTrue(segs[1]["data"]["file"].startswith("base64://"))

    def test_url_image_kept_as_is(self):
        segs = build_message("", ["https://x/a.gif"])
        self.assertEqual(segs, [{"type": "image", "data": {"file": "https://x/a.gif"}}])

    def test_rejected_image_dropped_but_text_kept(self):
        segs = build_message("只有文字", ["C:/not/allowed/a.png"], allow_dirs=[])
        self.assertEqual(segs, [{"type": "text", "data": {"text": "只有文字"}}])

    def test_max_images_cap(self):
        segs = build_message("", ["https://x/1.gif", "https://x/2.gif"],
                             max_images=1)
        self.assertEqual(len(segs), 1)


class TestStickerStore(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(os.path.join(self._tmp.name, "t.db"))

    async def asyncTearDown(self):
        await self.store.close()
        self._tmp.cleanup()

    async def test_add_get_search(self):
        sid = await self.store.add_sticker("h1", file="a.png", description="开心地笑",
                                          tags="开心 笑", emotion="开心")
        row = await self.store.get_sticker("h1")
        self.assertEqual(row["id"], sid)
        self.assertEqual(row["description"], "开心地笑")
        hits = await self.store.search_stickers("开心")
        self.assertEqual([h["hash"] for h in hits], ["h1"])

    async def test_add_is_idempotent(self):
        await self.store.add_sticker("h1", description="第一次")
        await self.store.add_sticker("h1", description="第二次")   # hash 冲突 → 忽略
        rows = await self.store.list_stickers()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["description"], "第一次")

    async def test_touch_and_delete(self):
        sid = await self.store.add_sticker("h1", description="无语")
        await self.store.touch_sticker(sid)
        row = await self.store.get_sticker("h1")
        self.assertEqual(row["use_count"], 1)
        self.assertTrue(await self.store.delete_sticker(sid))
        self.assertIsNone(await self.store.get_sticker("h1"))

    async def test_update_sticker_partial_fields(self):
        """后台编辑只改传入的字段，其余保持原值（None ≠ 空串）。"""
        sid = await self.store.add_sticker("h1", description="原来的描述",
                                           tags="old", emotion="无语")
        self.assertTrue(await self.store.update_sticker(sid, tags="新标签 修正"))
        row = await self.store.get_sticker_by_id(sid)
        self.assertEqual(row["tags"], "新标签 修正")
        self.assertEqual(row["description"], "原来的描述")   # 没传的字段不被清空
        self.assertEqual(row["emotion"], "无语")
        self.assertFalse(await self.store.update_sticker(sid))   # 什么都不传 = 无操作

    async def test_search_prefers_tags_over_description(self):
        """tag 命中权重必须高于描述模糊匹配：人工修正过的标签才是准的。"""
        await self.store.add_sticker("desc", description="开心地笑",
                                     tags="别的词")
        await self.store.add_sticker("tag", description="一只猫",
                                     tags="开心 猫")
        hits = await self.store.search_stickers("开心")
        self.assertEqual(hits[0]["hash"], "tag")

    async def test_search_tag_and_operator(self):
        """a+b 表示同一组内必须全部命中；空格表示 OR 候选。"""
        await self.store.add_sticker("both", description="猫在笑",
                                     tags="开心 猫")
        await self.store.add_sticker("only", description="只是开心",
                                     tags="开心")
        hits = await self.store.search_stickers("开心+猫")
        self.assertEqual([h["hash"] for h in hits], ["both"])
        # OR：两个都能命中，但 AND 命中的那个得分更高排前面
        hits = await self.store.search_stickers("开心 猫")
        self.assertEqual(hits[0]["hash"], "both")
        self.assertEqual({h["hash"] for h in hits}, {"both", "only"})


class TestStickerBook(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(os.path.join(self._tmp.name, "t.db"))
        self.vision = FakeVision()
        self.fetched = 0

        async def fake_fetch(url: str, max_bytes: int) -> bytes | None:
            self.fetched += 1
            return PNG_BYTES

        self.book = StickerBook(
            self.store, {"sticker_dir": os.path.join(self._tmp.name, "stickers")},
            self.vision, fetch=fake_fetch)

    async def asyncTearDown(self):
        await self.store.close()
        self._tmp.cleanup()

    async def test_first_time_parses_and_saves(self):
        note = await self.book.analyze({"kind": "sticker", "url": "http://x/a.png"})
        self.assertEqual(note, "开心地笑")
        self.assertEqual(self.vision.calls, 1)
        rows = await self.store.list_stickers()
        self.assertEqual(len(rows), 1)
        self.assertTrue(os.path.isfile(rows[0]["file"]))     # 表情包已落盘

    async def test_same_content_skips_vision(self):
        """同一个人重发同一个表情（URL 带新签名）不应再跑一次视觉。"""
        await self.book.analyze({"kind": "sticker", "url": "http://x/a.png?sign=1"})
        note = await self.book.analyze({"kind": "sticker", "url": "http://x/a.png?sign=2"})
        self.assertEqual(note, "开心地笑")
        self.assertEqual(self.vision.calls, 1)               # 只解析过一次
        self.assertEqual(self.fetched, 2)                    # 但两次都下载了（内容哈希去重）

    async def test_plain_photo_not_stored(self):
        note = await self.book.analyze({"kind": "image", "url": "http://x/b.png"})
        self.assertEqual(note, "开心地笑")
        self.assertEqual(await self.store.list_stickers(), [])   # 普通照片不入库

    async def test_face_without_url_uses_summary(self):
        note = await self.book.analyze({"kind": "face", "id": "178", "summary": "[呲牙]"})
        self.assertEqual(note, "[呲牙]")
        self.assertEqual(self.vision.calls, 0)               # 内置表情没有图可解析

    async def test_pick_by_query(self):
        await self.book.analyze({"kind": "sticker", "url": "http://x/a.png"})
        row = await self.book.pick("开心")
        self.assertIsNotNone(row)
        self.assertTrue(row["file"].endswith(".png"))
        self.assertIsNone(await self.book.pick("完全不相关的词"))

    # ---------------------------------------------------------------- 后台管理
    async def test_import_one_dedupes_by_content(self):
        """批量导入按 sha256 去重：同一张图第二次只算 skipped，不重复跑视觉。"""
        first = await self.book.import_one(PNG_BYTES, name="a.png")
        second = await self.book.import_one(PNG_BYTES, name="b.png")
        self.assertEqual(first["status"], "added")
        self.assertEqual(second["status"], "skipped")
        self.assertEqual(self.vision.calls, 1)
        rows = await self.store.list_stickers()
        self.assertEqual(len(rows), 1)
        self.assertTrue(os.path.isfile(rows[0]["file"]))

    async def test_import_one_uses_filename_when_no_vision(self):
        """没有视觉能力时退回文件名，至少不是空描述（否则永远检索不到）。"""
        self.book.vision = None
        result = await self.book.import_one(PNG_BYTES, name="无语猫.png")
        self.assertEqual(result["status"], "added")
        self.assertEqual(result["description"], "无语猫")

    async def test_update_and_delete_removes_file(self):
        result = await self.book.import_one(PNG_BYTES, name="a.png")
        sid = result["id"]
        self.assertTrue(await self.book.update(sid, description="手动改的描述",
                                               tags="无语 猫"))
        row = await self.store.get_sticker_by_id(sid)
        self.assertEqual(row["description"], "手动改的描述")
        self.assertEqual(row["tags"], "无语 猫")
        path = row["file"]
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(await self.book.delete(sid))
        self.assertFalse(os.path.exists(path))          # 文件随记录一起删掉
        self.assertIsNone(await self.store.get_sticker_by_id(sid))

    async def test_image_only_reads_inside_sticker_dir(self):
        sid = (await self.book.import_one(PNG_BYTES, name="a.png"))["id"]
        got = await self.book.image(sid)
        self.assertIsNotNone(got)
        self.assertEqual(got[1], "image/png")
        # 把 file 指到表情包目录外：必须拒绝（防后台接口变成任意文件读取）
        outside = os.path.join(self._tmp.name, "outside.png")
        with open(outside, "wb") as fh:
            fh.write(PNG_BYTES)
        await self.store.update_sticker(sid, file=outside)
        self.assertIsNone(await self.book.image(sid))


class TestQQSegments(unittest.TestCase):
    def test_extract_images_and_text(self):
        ev = {"message": [
            {"type": "text", "data": {"text": "看这个"}},
            {"type": "image", "data": {"url": "http://x/a.png", "sub_type": 1,
                                       "summary": "[表情包]"}},
            {"type": "face", "data": {"id": "178", "summary": "[呲牙]"}},
        ]}
        self.assertEqual(QQChannel._extract_text(ev), "看这个[表情包][表情]")
        images = QQChannel._extract_images(ev)
        self.assertEqual(images[0]["kind"], "sticker")
        self.assertEqual(images[0]["url"], "http://x/a.png")
        self.assertEqual(images[1]["kind"], "face")
        self.assertEqual(images[1]["id"], "178")


class _FakeCfg:
    """只实现 QQChannel 会用到的 section()。"""

    def __init__(self, sections: dict) -> None:
        self._sections = sections

    def section(self, name: str) -> dict:
        return dict(self._sections.get(name, {}))


class _FakeDeps:
    def __init__(self, sections: dict) -> None:
        self.bus = None
        self.settings = None
        self.cfg = _FakeCfg(sections)


class _FakeClient:
    """记录真正交给 OneBot 的 message 参数。"""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, object]] = []

    async def send_private_msg(self, user_id, message):
        self.sent.append(("private", str(user_id), message))
        return {}

    async def send_group_msg(self, group_id, message):
        self.sent.append(("group", str(group_id), message))
        return {}


class TestQQSendWithImages(unittest.IsolatedAsyncioTestCase):
    async def test_text_only_keeps_string_payload(self):
        """没有图时仍走原来的纯字符串路径（向后兼容）。"""
        ch = QQChannel(_FakeDeps({"qq": {}}))
        ch.client = _FakeClient()
        from ska.kernel.packet import CoreOutput
        ok = await ch.send(CoreOutput(content="你好", target="private_msg", user="1"))
        self.assertTrue(ok)
        self.assertEqual(ch.client.sent[0][2], "你好")

    async def test_url_image_becomes_segment(self):
        ch = QQChannel(_FakeDeps({"qq": {}, "media": {}}))
        ch.client = _FakeClient()
        from ska.kernel.packet import CoreOutput
        ok = await ch.send(CoreOutput(content="给你", target="private_msg", user="1",
                                      images=["https://x/a.gif"]))
        self.assertTrue(ok)
        message = ch.client.sent[0][2]
        self.assertIsInstance(message, list)
        self.assertEqual(message[1], {"type": "image", "data": {"file": "https://x/a.gif"}})

    async def test_sticker_file_in_sticker_dir_is_sendable(self):
        """表情包目录无需写进 allow_dirs，也应能发出去（send_dirs 自动并入）。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "s.png")
            with open(path, "wb") as fh:
                fh.write(PNG_BYTES)
            ch = QQChannel(_FakeDeps({"qq": {}, "media": {"sticker_dir": tmp}}))
            ch.client = _FakeClient()
            from ska.kernel.packet import CoreOutput
            await ch.send(CoreOutput(content="", target="private_msg", user="1",
                                     images=[path]))
            message = ch.client.sent[0][2]
        self.assertEqual(message[0]["type"], "image")
        self.assertTrue(message[0]["data"]["file"].startswith("base64://"))


class TestDecisionParsing(unittest.TestCase):
    def setUp(self):
        self.mind = Mind(None, None)          # 只测纯解析逻辑，不需要真的网关/执行者

    def test_sticker_and_images_fields(self):
        decision = self.mind._to_decision({
            "action": "sticker", "sticker": "开心", "images": ["E:/a.png"],
            "content": "哈哈",
        })
        self.assertEqual(decision.action, "sticker")
        self.assertEqual(decision.sticker, "开心")
        self.assertEqual(decision.images, ["E:/a.png"])

    def test_unknown_action_falls_back_to_speak(self):
        self.assertEqual(self.mind._to_decision({"action": "dance"}).action, "speak")

    def test_describe_merges_image_notes(self):
        from ska.kernel.packet import CoreInput
        trigger = CoreInput(source="qq_message", content="[表情包]", user="u1",
                            channel="qq")
        desc = self.mind._describe(trigger, ["开心地笑"])
        self.assertIn("开心地笑", desc)


class TestStillFrame(unittest.TestCase):
    """动图抽帧：视觉接口只认静态图，GIF 必须先转成一张 PNG。"""

    def test_non_animated_returns_original(self):
        data, mime = still_frame(PNG_BYTES, "image/png")
        self.assertEqual((data, mime), (PNG_BYTES, "image/png"))

    def test_mime_unknown_defaults_to_jpeg(self):
        self.assertEqual(still_frame(b"", "")[1], "image/jpeg")

    def test_gif_becomes_single_png(self):
        try:
            gif = _gif_bytes()
        except ImportError:
            self.skipTest("未安装 Pillow，无法生成 GIF 测试数据")
        data, mime = still_frame(gif, "image/gif")
        self.assertEqual(mime, "image/png")                # 改送 PNG，接口才收
        self.assertTrue(data.startswith(b"\x89PNG"))       # 确实是 PNG 魔数

    def test_corrupt_gif_falls_back(self):
        """解码失败不能抛异常，否则整条来图解析都会挂掉。"""
        raw = b"GIF89a-broken"
        data, mime = still_frame(raw, "image/gif")
        self.assertEqual((data, mime), (raw, "image/gif"))


class _RecordingVision:
    """记录每次解析收到的字节与 MIME，用来断言「送的是抽帧后的 PNG」。"""

    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []

    async def describe(self, data: bytes, mime: str = "image/jpeg") -> dict:
        self.calls.append((data, mime))
        return {"description": "动图首帧", "tags": "动图 搞笑",
                "emotion": "开心", "is_sticker": True}


class TestAnimatedSticker(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(os.path.join(self._tmp.name, "t.db"))
        self.vision = _RecordingVision()
        self.gif = _gif_bytes() if _has_pillow() else b""

        async def fake_fetch(url: str, max_bytes: int) -> bytes | None:
            return self.gif or None

        self.book = StickerBook(
            self.store, {"sticker_dir": os.path.join(self._tmp.name, "stickers")},
            self.vision, fetch=fake_fetch)

    async def asyncTearDown(self):
        await self.store.close()
        self._tmp.cleanup()

    async def test_gif_is_sent_as_png(self):
        if not self.gif:
            self.skipTest("未安装 Pillow，无法生成 GIF 测试数据")
        note = await self.book.analyze({"kind": "sticker", "url": "http://x/a.gif"})
        self.assertEqual(note, "动图首帧")
        data, mime = self.vision.calls[0]
        self.assertEqual(mime, "image/png")                # 不再写死 jpeg
        self.assertTrue(data.startswith(b"\x89PNG"))       # 送出去的是首帧


def _has_pillow() -> bool:
    try:
        import PIL  # noqa: F401
        return True
    except ImportError:
        return False


class _ScriptedGateway:
    """按脚本返回一次 chat 结果的网关替身。"""

    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.content = content
        self.finish_reason = finish_reason
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        return LLMResponse(content=self.content, model="fake",
                           finish_reason=self.finish_reason)


class TestVisionFallback(unittest.IsolatedAsyncioTestCase):
    """视觉输出的三种容错路径：正常 JSON / 空回复 / 非 JSON 纯文本。"""

    async def test_json_output_parsed(self):
        gw = _ScriptedGateway('{"description":"笑","tags":["笑","开心"],'
                              '"emotion":"开心","is_sticker":true}')
        out = await VisionDescriber(gw, {}).describe(PNG_BYTES)
        self.assertEqual(out["description"], "笑")
        self.assertEqual(out["tags"], "笑 开心")
        self.assertTrue(out["is_sticker"])

    async def test_empty_content_returns_empty(self):
        """空回复（多半被 max_tokens 截断）返回 {}，但要把默认预算调大。"""
        out = await VisionDescriber(_ScriptedGateway(""), {}).describe(PNG_BYTES)
        self.assertEqual(out, {})
        self.assertEqual(VisionDescriber(None, {}).max_tokens, 1200)

    async def test_non_json_text_degrades_to_description(self):
        """模型没按 JSON 输出时，把纯文本当描述，别整条丢弃。"""
        gw = _ScriptedGateway("```\n一只柴犬  歪头\n看着你\n```")
        out = await VisionDescriber(gw, {}).describe(PNG_BYTES)
        self.assertEqual(out["description"], "一只柴犬 歪头 看着你")   # 围栏与换行已净化
        self.assertEqual(out["tags"], "")
        self.assertFalse(out["is_sticker"])


class _ScriptedMindLLM:
    """Mind 用的脚本 LLM：按顺序弹「独白 → 决策」。"""

    def __init__(self, script: list[tuple[str, object]]) -> None:
        self.script = script
        self.i = 0

    async def chat(self, messages, *, tools=None, json_mode=False,
                   temperature=None, max_tokens=None, task="") -> LLMResponse:
        idx = min(self.i, len(self.script) - 1)
        self.i += 1
        text, tool = self.script[idx]
        return LLMResponse(content=text, tool_call=tool, model="fake")


class _StickerOnlyBook:
    """永远能命中表情包的库替身（list 给决策提示词，pick 给实际发送的文件）。"""

    ROW = {"hash": "h1", "file": "E:/s.png", "description": "无语"}

    async def list(self, limit: int = 100) -> list[dict]:
        return [self.ROW]

    async def pick(self, query: str, *, user: str = "") -> dict:
        return self.ROW


class _EmptyStickerBook:
    """库里没有任何表情包：决策选 sticker 时必须退化回说话。"""

    async def list(self, limit: int = 100) -> list[dict]:
        return []

    async def pick(self, query: str, *, user: str = "") -> None:
        return None


class TestStickerOutput(unittest.IsolatedAsyncioTestCase):
    """action=sticker 的出口语义：命中就只发表情包，不附带任何文字。"""

    def _mind(self, book) -> Mind:
        llm = _ScriptedMindLLM([
            ('{"action":"sticker","sticker":"无语","content":"别闹了哈","importance":0.3}',
             None),
        ])
        return Mind(llm, AgentLoop(llm, {}, "你是助手"), stickers=book,
                    persona_prompt="你叫小赛。")

    async def test_sticker_hit_sends_image_without_text(self):
        result = await self._mind(_StickerOnlyBook()).wake(
            SessionContext(20),
            CoreInput(source="qq_message", content="哈哈哈", user="683t", channel="qq"))
        self.assertEqual(result.decision.action, "sticker")
        self.assertEqual(result.output.content, "")          # 配文被丢弃
        self.assertEqual(result.output.images, ["E:/s.png"])  # 只留表情包

    async def test_sticker_miss_degrades_to_speak_with_text(self):
        result = await self._mind(_EmptyStickerBook()).wake(
            SessionContext(20),
            CoreInput(source="qq_message", content="哈哈哈", user="683t", channel="qq"))
        self.assertEqual(result.decision.action, "speak")
        self.assertEqual(result.output.content, "别闹了哈")   # 兜底文案这时才用上
        self.assertEqual(result.output.images, [])


class _CountingBook:
    """只统计「解析了几张图」的表情包库替身。"""

    def __init__(self) -> None:
        self.calls = 0

    async def analyze(self, image: dict, *, user: str = "", channel: str = "") -> str:
        self.calls += 1
        return f"第{self.calls}张图"

    async def list(self, limit: int = 100) -> list[dict]:
        return []


class _MediaCfg:
    """Mind 需要的最小配置替身（只关心 media 小节）。"""

    def __init__(self, media: dict) -> None:
        self.memory: dict = {}
        self.llm: dict = {}
        self._media = media

    def section(self, name: str) -> dict:
        return self._media if name == "media" else {}


class TestBatchImageCapture(unittest.IsolatedAsyncioTestCase):
    """对话窗口合批后，窗口内每张表情包都要被解析入库（不能被单条上限截掉）。"""

    async def test_batch_raises_image_limit(self):
        book = _CountingBook()
        mind = Mind(_ScriptedMindLLM([]), None,
                    cfg=_MediaCfg({"max_images_per_message": 2}), stickers=book)
        images = [{"kind": "sticker", "url": f"http://x/{i}.gif"} for i in range(5)]

        # 单条消息：仍按 max_images_per_message=2 截断（防止刷图拖垮一轮推理）
        single = CoreInput(source="qq_message", content="看", images=images)
        await mind._analyze_images(single, trace_id="t1", session="s", sandbox=False)
        self.assertEqual(book.calls, 2)

        # 合批（4 条消息）：上限按批大小放宽 → 5 张全部解析，后几张才不会漏掉
        book.calls = 0
        batch = CoreInput(source="qq_message", content="看", images=images,
                          raw={"_batch": 4})
        await mind._analyze_images(batch, trace_id="t2", session="s", sandbox=False)
        self.assertEqual(book.calls, 5)

    async def test_batch_limit_has_absolute_cap(self):
        """再大的批也不能无限解析：MAX_BATCH_IMAGES 兜底。"""
        from ska.loop.mind import MAX_BATCH_IMAGES
        book = _CountingBook()
        mind = Mind(_ScriptedMindLLM([]), None,
                    cfg=_MediaCfg({"max_images_per_message": 4}), stickers=book)
        images = [{"kind": "sticker", "url": f"http://x/{i}.gif"}
                  for i in range(MAX_BATCH_IMAGES + 10)]
        batch = CoreInput(source="qq_message", content="刷图", images=images,
                          raw={"_batch": 99})
        await mind._analyze_images(batch, trace_id="t3", session="s", sandbox=False)
        self.assertEqual(book.calls, MAX_BATCH_IMAGES)


if __name__ == "__main__":
    unittest.main()
