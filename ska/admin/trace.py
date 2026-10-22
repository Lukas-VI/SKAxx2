"""TraceLog —— MindLine 轨迹的落库与广播中枢（阶段三，见方案 §4.3）。

决策 E4 的落地：**trace 是 log 的一部分**。因此本类的每一次写入都做两件事：
    1. 落库 admin.db 的 traces 表（供 MindLine 页面按 seq/会话增量查询）；
    2. 走既有的 `logging`（`ska.trace` 这个 logger），于是它自然汇入
       main.py 里 basicConfig 配置的日志流 —— 不额外引入日志系统。

另外维护一个内存环形缓冲（deque，每 trace 最近 N 条），让页面在不查库、
或库不可用时也能立刻读到最新轨迹（与阶段一/二「无后台也能跑」的风格一致）。

并发：seq 由本类统一分配（全局单调递增），前端 `?after=seq` 增量拉取。
分配用的自增计数放在 threading.Lock 里保护，避免多线程（Web 线程）读到半值——
虽然写入本身在事件循环里，但 recent()/page 查询可能来自 Web 线程。
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any

from ..kernel.event import TOPIC_TRACE, Event
from ..kernel.packet import TraceEvent

logger = logging.getLogger("ska.trace")


async def emit_trace(bus: Any, *, trace_id: str, session: str, lane: str, kind: str,
                     title: str = "", detail: str = "", tokens: int = 0,
                     sandbox: bool = False) -> Any:
    """往总线发一条轨迹事件（供 Mind / AgentLoop 插桩调用）。

    bus 为 None 时直接返回 None —— 这样没有总线（如单测构造 Mind 时未注入）
    也能正常跑，插桩对业务逻辑零侵入。
    """
    if bus is None:
        return None
    try:
        event = TraceEvent(trace_id=trace_id, session=session, lane=lane,  # type: ignore[arg-type]
                           kind=kind, title=title, detail=detail,
                           tokens=tokens, sandbox=sandbox)
    except Exception as e:  # noqa: BLE001
        # 轨迹是**旁路观测**：一个拼错的 kind/lane 不该让整轮对话被跳过。
        # 这里降级为「这条轨迹丢掉 + 告警」，与 TraceLog.write 的落库容错同一原则。
        logger.warning("轨迹事件不合法，已跳过（lane=%s, kind=%s）：%s",
                       lane, kind, e)
        return None
    await bus.emit(TOPIC_TRACE, event)
    return event


class TraceLog:
    """轨迹落库 + 环形缓冲。"""

    def __init__(self, store: Any = None, *, buffer: int = 500) -> None:
        self.store = store                              # AdminStore（可为 None：只进日志）
        self.buffer_size = max(int(buffer), 1)
        self._buffer: deque[dict[str, Any]] = deque(maxlen=self.buffer_size)
        self._seq = 0                                   # 全局递增序号（由本类分配）
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ 订阅
    def attach(self, bus: Any) -> None:
        """订阅总线上的 TOPIC_TRACE（信号类事件，不进入主循环推理）。

        与插件订阅同一套总线机制：Mind/Agent 只负责 emit 轨迹，
        谁来存储、谁来展示与本类解耦（阶段三把二者合并到了这里）。
        """
        bus.subscribe(TOPIC_TRACE, self.on_event)

    async def on_event(self, event: Event) -> None:
        """总线回调：一条 TraceEvent 载荷 → 落库 + 日志 + 入环。"""
        await self.write(event.payload)

    # ------------------------------------------------------------------ 写入
    async def write(self, payload: Any) -> dict[str, Any]:
        """处理一条轨迹载荷（TraceEvent 或任意带同名属性的对象）。

        返回归一化后的 dict（测试与页面都按它断言/渲染）。
        """
        seq = self._next_seq()
        item = self._as_dict(payload, seq)
        self._buffer.append(item)
        # ① 走既有 logger（决策 E4：trace 是 log 的一部分）
        logger.info("[trace] %s | %s/%s | %s | %s",
                    item["session"] or "-", item["lane"], item["kind"],
                    item["title"], item["detail"][:200])
        # ② 落库（库不可用时降级为「只进日志」，绝不打断推理）
        if self.store is not None:
            try:
                await self.store.add_trace(
                    trace_id=item["trace_id"], seq=seq, ts=item["ts"],
                    session=item["session"], lane=item["lane"], kind=item["kind"],
                    title=item["title"], detail=item["detail"],
                    tokens=item["tokens"], sandbox=item["sandbox"])
            except Exception:  # noqa: BLE001 —— 旁路观测失败不影响主链
                logger.exception("轨迹落库失败（已仅保留内存缓冲）")
        return item

    # ------------------------------------------------------------------ 查询
    def recent(self, after: int = 0, session: str = "", limit: int = 300) -> list[dict[str, Any]]:
        """从内存环形缓冲增量读取（前端 1 秒轮询用，不查库、够快）。"""
        out = [x for x in self._buffer
               if int(x["seq"]) > int(after) and (not session or x["session"] == session)]
        return out[-max(int(limit), 1):]

    def last_seq(self) -> int:
        """当前已分配的最大序号（前端游标初始化）。"""
        with self._lock:
            return self._seq

    def clear(self) -> None:
        """清空内存环形缓冲（后台「清空轨迹」时与清库一起调用）。"""
        self._buffer.clear()

    # ------------------------------------------------------------------ 内部
    def _next_seq(self) -> int:
        with self._lock:
            self._seq += 1
            return self._seq

    @staticmethod
    def _as_dict(payload: Any, seq: int) -> dict[str, Any]:
        """把载荷归一化成 dict（缺字段安全兜底，兼容测试替身）。"""
        return {
            "seq": seq,
            "trace_id": str(getattr(payload, "trace_id", "") or ""),
            "ts": float(getattr(payload, "ts", 0.0) or 0.0),
            "session": str(getattr(payload, "session", "") or ""),
            "lane": str(getattr(payload, "lane", "think") or "think"),
            "kind": str(getattr(payload, "kind", "monologue") or "monologue"),
            "title": str(getattr(payload, "title", "") or ""),
            "detail": str(getattr(payload, "detail", "") or ""),
            "tokens": int(getattr(payload, "tokens", 0) or 0),
            "sandbox": bool(getattr(payload, "sandbox", False)),
        }
