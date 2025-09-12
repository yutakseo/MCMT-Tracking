# /workspace/MCMT_engine/core/online_projection.py
from __future__ import annotations
import threading, queue, time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

# PlanProjector 는 당신의 homoGraphy.py 최적화 버전 import
from MCMT_engine.core.homoGraphy import PlanProjector

@dataclass
class DetItem:
    frame_id: int
    boxes_xyxy: np.ndarray  # (N,4) float32
    ids: Optional[np.ndarray] = None  # (N,) int or None
    clss: Optional[List[str]] = None  # (N,) list[str] or None
    meta: Optional[Dict[str, Any]] = None

@dataclass
class ProjItem:
    frame_id: int
    pts_plan: np.ndarray      # (N,2) float32
    boxes_xyxy: np.ndarray    # (N,4) float32 (원본 유지; 필요 시 오버레이용)
    ids: Optional[np.ndarray] = None
    clss: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None


class OnlineProjector:
    """
    디텍션 결과가 '프레임 단위'로 나오자마자 즉시 투영하는 워커.
    - det_q → (투영) → proj_q
    - 순서보장을 위해 consumer 쪽에서 ReorderBuffer 사용
    """
    def __init__(
        self,
        projector: PlanProjector,
        det_q: queue.Queue,
        proj_q: queue.Queue,
        stop_event: threading.Event,
        warmup_spin: float = 0.0,
    ):
        self.proj = projector
        self.det_q = det_q
        self.proj_q = proj_q
        self.stop = stop_event
        self.warmup_spin = warmup_spin
        self._th: Optional[threading.Thread] = None

    def start(self) -> None:
        self._th = threading.Thread(target=self._run, name="OnlineProjector", daemon=True)
        self._th.start()

    def join(self, timeout: Optional[float] = None) -> None:
        if self._th:
            self._th.join(timeout=timeout)

    def _run(self):
        if self.warmup_spin > 0:
            time.sleep(self.warmup_spin)

        while not self.stop.is_set():
            try:
                item: DetItem = self.det_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Fast path: 바로 투영
            if item.boxes_xyxy.size == 0:
                proj = np.empty((0, 2), dtype=np.float32)
            else:
                proj = self.proj.projection_from_xyxy(item.boxes_xyxy)

            out = ProjItem(
                frame_id=item.frame_id,
                pts_plan=proj,
                boxes_xyxy=item.boxes_xyxy,
                ids=item.ids,
                clss=item.clss,
                meta=item.meta
            )
            # backpressure: proj_q가 가득이면 여기서 자연스레 대기 → 상류 속도 조절
            self.proj_q.put(out)


class ReorderEmitter:
    """
    프레임 순서 보장용 소형 리오더 버퍼.
    proj_q에서 오는 ProjItem을 frame_id 순서에 맞춰 sink로 전달.
    """
    def __init__(self, start_frame_id: int, emit_fn, max_hold: int = 64):
        self.next_to_emit = start_frame_id
        self.buf: Dict[int, ProjItem] = {}
        self.emit_fn = emit_fn
        self.max_hold = max_hold

    def offer(self, item: ProjItem):
        fid = item.frame_id
        if fid == self.next_to_emit:
            self.emit_fn(item)
            self.next_to_emit += 1
            # 누적 정렬 방출
            while self.next_to_emit in self.buf:
                it = self.buf.pop(self.next_to_emit)
                self.emit_fn(it)
                self.next_to_emit += 1
        else:
            if len(self.buf) < self.max_hold:
                self.buf[fid] = item
            else:
                # 버퍼 과다 시 지연된 프레임 드랍 (정책 선택)
                # 여기서는 가장 오래된 것 하나 제거
                oldest_key = min(self.buf.keys())
                self.buf.pop(oldest_key, None)
                self.buf[fid] = item


class OnlineProjectionPipeline:
    """
    외부(Tracker/Detector)에서 frame 단위 디텍션 결과를 push 하면,
    즉시 투영하고 순서에 맞게 sink로 넘김.
    """
    def __init__(
        self,
        projector: PlanProjector,
        max_queue: int = 64,
    ):
        self.det_q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.proj_q: queue.Queue = queue.Queue(maxsize=max_queue)
        self.stop = threading.Event()
        self.worker = OnlineProjector(projector, self.det_q, self.proj_q, self.stop)

    def start(self) -> None:
        self.worker.start()

    def stop_and_join(self) -> None:
        self.stop.set()
        self.worker.join(timeout=2.0)

    # 외부에서 프레임 단위로 호출: 디텍터가 배치 완료 즉시 각 프레임을 개별 push
    def on_detection(
        self,
        frame_id: int,
        boxes_xyxy: np.ndarray,
        ids: Optional[np.ndarray] = None,
        clss: Optional[List[str]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        # float32 보장
        boxes = np.asarray(boxes_xyxy, dtype=np.float32, order="C")
        self.det_q.put(DetItem(frame_id, boxes, ids, clss, meta))

    # proj_q 소비 루프: 순서 보장 후 sink로 전달
    def consume_and_emit(self, emit_fn, start_frame_id: int = 0, poll_timeout: float = 0.1):
        reorder = ReorderEmitter(start_frame_id, emit_fn)
        while not self.stop.is_set():
            try:
                item: ProjItem = self.proj_q.get(timeout=poll_timeout)
            except queue.Empty:
                continue
            reorder.offer(item)
