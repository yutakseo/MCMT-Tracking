# /workspace/MCMT_engine/core/stream_track_encode.py
import os, time, gc, cv2
import numpy as np
from typing import Callable, Iterable, Tuple, List, Dict, Any, Optional
from __Tracking.tracking_api import TrackerAPI

OnDetection = Callable[[int, np.ndarray, Optional[np.ndarray], Optional[List[str]], Optional[dict]], None]

class StreamTrackEncode:
    def __init__(self, tracker: TrackerAPI, *, batch_size=20, infer_timeout=0.20, min_flush=None):
        self.tracker = tracker
        self.batch_size = int(batch_size)
        self.infer_timeout = float(infer_timeout)
        self.min_flush = int(min_flush) if min_flush is not None else max(2, self.batch_size // 2)

    @staticmethod
    def _build_det_arrays(frame_res: List[Dict[str, Any]]):
        if not frame_res:
            return np.empty((0,4), np.float32), None, None

        def _xyxy_from_any(b):
            v = np.asarray(b, np.float32).reshape(-1)
            if v.shape[0] < 4:
                raise ValueError("bbox length must be >= 4")
            x1, y1, a, b2 = v[0], v[1], v[2], v[3]
            # case1: 이미 xyxy (우상단이 더 큼)
            if a > x1 and b2 > y1:
                x2, y2 = a, b2
            # case2: tlwh
            elif a >= 0 and b2 >= 0:
                x2, y2 = x1 + a, y1 + b2
            else:
                # 뒤바뀐 좌표 방지
                x2, y2 = a, b2
            # 정렬
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            return np.array([x1, y1, x2, y2], np.float32)

        boxes, ids, clss = [], [], []
        for d in frame_res:
            b = d.get("bbox", None)
            if b is None:
                b = d.get("box", None)
            if b is None:
                b = d.get("tlwh", None)
            if b is None:
                continue

            boxes.append(_xyxy_from_any(b))
            ids.append(d.get("id"))
            clss.append(d.get("label", d.get("class", None)))

        boxes_xyxy = np.vstack(boxes) if boxes else np.empty((0,4), np.float32)
        ids_arr = (np.array([(-1 if v is None else int(v)) for v in ids], np.int32)
                if any(v is not None for v in ids) else None)
        cls_list = ([("" if v is None else str(v)) for v in clss]
                    if any(v is not None for v in clss) else None)
        return boxes_xyxy, ids_arr, cls_list


    def run_stream(
        self,
        stream_fn: Callable[[str], Iterable[Tuple[int, np.ndarray]]],
        video_path: str,
        *,
        on_detection: Optional[OnDetection] = None,
        visualize_fn: Optional[Callable[[np.ndarray, List[Dict[str, Any]]], np.ndarray]] = None,
        save_path: Optional[str] = None,
        writer_size: Optional[Tuple[int,int]] = None,
        writer_fps: float = 30.0,
    ) -> Dict[int, List[Dict[str, Any]]]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        self.tracker.reset()
        results: List[List[Dict[str, Any]]] = []

        writer = None
        if save_path:
            w, h = writer_size or self._probe_size(video_path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            writer = cv2.VideoWriter(save_path, fourcc, writer_fps, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"VideoWriter open failed: {save_path}")

        stream = stream_fn(video_path)
        batch_frames, batch_raw, batch_fids = [], [], []
        last_infer_t = time.perf_counter()

        try:
            for fid, frame in stream:
                batch_frames.append(frame)
                if writer is not None or visualize_fn is not None:
                    batch_raw.append(frame)
                batch_fids.append(fid)

                now = time.perf_counter()
                do_flush = (len(batch_frames) >= self.batch_size) or (
                    self.infer_timeout > 0 and (now - last_infer_t) >= self.infer_timeout and len(batch_frames) >= self.min_flush
                )
                if not do_flush:
                    continue

                batch_res = self.tracker.track_batch(batch_frames)

                if on_detection is not None:
                    for _fid, _res in zip(batch_fids, batch_res):
                        b, i, c = self._build_det_arrays(_res)
                        on_detection(_fid, b, i, c, None)

                if writer is not None or visualize_fn is not None:
                    for f, r in zip(batch_raw, batch_res):
                        out = visualize_fn(f, r) if visualize_fn else f
                        if writer is not None:
                            writer.write(out)
                        results.append(r)

                batch_frames.clear(); batch_raw.clear(); batch_fids.clear()
                last_infer_t = now
                del batch_res
                gc.collect()
        finally:
            if writer is not None:
                writer.release()

        return {i: fr for i, fr in enumerate(results)}

    @staticmethod
    def _probe_size(video_path: str) -> Tuple[int,int]:
        cap = cv2.VideoCapture(video_path)
        ok = cap.isOpened()
        w, h = int(cap.get(3)), int(cap.get(4)) if ok else (0,0)
        cap.release()
        if w <= 0 or h <= 0:
            raise RuntimeError("Cannot probe video size")
        return (w, h)
