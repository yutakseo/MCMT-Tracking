# /workspace/__Tracking/tracking_api.py
import cv2, os, gc, time
import numpy as np
from typing import List, Dict, Any, Iterable, Tuple, Optional, Callable

from __Tracking.core.tracker_core import TrackerCore
from __Tracking.utils.visualizer import TrackerVisualizer

# 새 스트리머 팩토리(디코더)는 MCMT의 core 모듈에서 가져온다
from MCMT_engine.core.video_stream import stream_factory_from_args


class Args:
    # 트래킹 파라미터
    track_thresh = 0.1
    match_thresh = 0.9
    track_buffer = 180
    mot20 = False

    # 디코더/배치 관련 (TrackerAPI는 단순히 값을 보관/로그만 함)
    cpu_workers = 10               # (구) 이름
    decode_threads = None          # (신) 이름, None이면 cpu_workers 사용
    prefetch_frames = 512
    hwaccel = None
    decode_target_size = None      # (w,h)

    batch_size = 20

    # 배치 플러시 정책 (새로 추가)
    infer_timeout = 0.20           # 초; 0이면 타임아웃 플러시 비활성
    min_flush = None               # None이면 max(2, batch_size//2)

    # 이전 호환(사용 안함)
    chunk_sec = 10.0


class TrackerAPI:
    def __init__(
        self,
        args: Optional[Args] = None,
        detector=None,
        streamer: Optional[Callable[[str], Iterable[Tuple[int, np.ndarray]]]] = None,
    ) -> None:
        """
        streamer: (video_path) -> iterator[(idx, frame)]
                  주입하지 않으면 MCMT_engine.core.video_stream.stream_factory_from_args(args)를 사용
        """
        if args is None:
            self.args = Args()
        else:
            self.args = args
        print("TrackingAPI activated...")

        if detector is None or not hasattr(detector, "detect"):
            raise ValueError("detector must be provided and implement a .detect(frame) method")

        self.core = TrackerCore(self.args, detector)

        # 스트리머 DI
        if streamer is None:
            self.streamer = stream_factory_from_args(self.args)
        else:
            self.streamer = streamer

        # 로그용 파라미터 추출(디코더 설정은 streamer.cfg가 보유)
        self.decode_threads = int(
            getattr(self.args, "decode_threads",
                    getattr(self.args, "cpu_workers", 0)) or 0
        )
        self.prefetch_frames = int(getattr(self.args, "prefetch_frames", 256))
        self.hwaccel = getattr(self.args, "hwaccel", None)
        self.decode_target_size: Optional[Tuple[int, int]] = getattr(self.args, "decode_target_size", None)

        self.visualizer = TrackerVisualizer()
        self.results: List[List[Dict[str, Any]]] = []
        self.batch_size = int(getattr(self.args, "batch_size", 8))

        # 플러시 정책
        self.infer_timeout = float(getattr(self.args, "infer_timeout", 0.20))
        _mf = getattr(self.args, "min_flush", None)
        self.min_flush = int(_mf) if _mf is not None else max(2, self.batch_size // 2)

        print(
            f"[TrackerAPI] __init__: decode_threads={self.decode_threads}, prefetch={self.prefetch_frames}, "
            f"hwaccel={self.hwaccel}, target_size={self.decode_target_size}, "
            f"batch_size={self.batch_size}, infer_timeout={self.infer_timeout}, min_flush={self.min_flush}"
        )

    @staticmethod
    def _pack_results(results: List[List[Dict[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
        return {i: frame_res for i, frame_res in enumerate(results)}

    # ────────────────────────────
    # 내부 유틸: bbox 표준화(xyxy)
    # ────────────────────────────
    @staticmethod
    def _xyxy_from_any(bbox: Any) -> np.ndarray:
        """
        bbox: [x1,y1,x2,y2] 또는 [x,y,w,h] 또는 (순서 뒤바뀐 좌표)
        return: (4,) float32 xyxy
        """
        b = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if b.shape[0] < 4:
            raise ValueError("bbox length must be >= 4")
        x1, y1, a, b2 = b[0], b[1], b[2], b[3]
        # 이미 xyxy
        if a > x1 and b2 > y1:
            return np.array([x1, y1, a, b2], dtype=np.float32)
        # tlwh
        if a > 0 and b2 > 0:
            return np.array([x1, y1, x1 + a, y1 + b2], dtype=np.float32)
        # 섞임 방지
        x2, y2 = a, b2
        return np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)

    def _build_det_arrays(self, frame_res: List[Dict[str, Any]]) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[List[str]]]:
        """
        frame_res(list[dict]) -> boxes_xyxy(float32 Nx4), ids(Optional N,), classes(Optional list[str])
        dict 스키마 예: {'bbox':..., 'id':..., 'label'/ 'class': ...}
        """
        if not frame_res:
            return np.empty((0, 4), dtype=np.float32), None, None

        boxes = []
        ids: List[Optional[int]] = []
        clss: List[Optional[str]] = []

        append = boxes.append
        for d in frame_res:
            bbox = d.get("bbox", d.get("box", d.get("tlwh", None)))
            if bbox is None:
                continue
            append(self._xyxy_from_any(bbox))
            ids.append(int(d["id"]) if "id" in d and d["id"] is not None else None)
            clss.append(d.get("label", d.get("class", None)))

        if not boxes:
            return np.empty((0, 4), dtype=np.float32), None, None

        boxes_xyxy = np.vstack(boxes).astype(np.float32, copy=False)

        # ids: Optional[np.ndarray] (정수/None 섞임을 허용 → object 배열 방지 위해 필터링)
        if any(v is not None for v in ids):
            ids_array = np.array([(-1 if v is None else int(v)) for v in ids], dtype=np.int32)
        else:
            ids_array = None

        # classes: Optional[List[str]]
        if any(v is not None for v in clss):
            class_list = [("" if v is None else str(v)) for v in clss]
        else:
            class_list = None

        return boxes_xyxy, ids_array, class_list

    # ----------------------------
    # 내부: 비디오 전체 추적 (저장 X)
    # ----------------------------
    def _track_video(self, video_path: str) -> Dict[int, List[Dict[str, Any]]]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path does not exist: {video_path}")

        self.results = []
        self.core.reset_tracker()
        self.core.img_size = None

        print(
            f"[TrackerAPI] decode(streaming): start "
            f"(threads={self.decode_threads}, prefetch={self.prefetch_frames}, "
            f"hwaccel={self.hwaccel}, target_size={self.decode_target_size})"
        )

        stream = self.streamer(video_path)  # (idx, frame) iterator

        batch_frames: List[np.ndarray] = []
        batch_fids: List[int] = []
        frame_in = 0
        print("[TrackerAPI] detect+track: start")
        t0_all = time.perf_counter()
        last_infer_t = time.perf_counter()

        for fid, frame in stream:
            batch_frames.append(frame)
            batch_fids.append(fid)
            now = time.perf_counter()

            do_flush = False
            if len(batch_frames) >= self.batch_size:
                do_flush = True
            elif self.infer_timeout > 0 and (now - last_infer_t) >= self.infer_timeout and len(batch_frames) >= self.min_flush:
                do_flush = True

            if do_flush:
                t0 = time.perf_counter()
                batch_res = self.core.track_video_batch(batch_frames)
                t1 = time.perf_counter()

                self.results.extend(batch_res)
                frame_in += len(batch_frames)
                last_infer_t = now

                if frame_in % max(1, (self.batch_size * 10)) == 0:
                    print(f"[TrackerAPI] detect+track: processed {frame_in} frames "
                          f"(batch={len(batch_frames)}, infer={(t1 - t0):.3f}s)")

                batch_frames.clear()
                batch_fids.clear()
                del batch_res
                gc.collect()

        if batch_frames:
            t0 = time.perf_counter()
            batch_res = self.core.track_video_batch(batch_frames)
            t1 = time.perf_counter()
            self.results.extend(batch_res)
            frame_in += len(batch_frames)
            print(f"[TrackerAPI] detect+track: processed {frame_in} frames "
                  f"(final, infer={(t1 - t0):.3f}s)")
            batch_frames.clear()
            batch_fids.clear()
            del batch_res
            gc.collect()

        t1_all = time.perf_counter()
        print(f"[TrackerAPI] detect+track: done frames={len(self.results)} time={(t1_all - t0_all):.3f}s")

        if not self.results:
            raise RuntimeError(f"No frames were processed from: {video_path}")
        return self._pack_results(self.results)

    # ----------------------------
    # 단일 프레임 추적
    # ----------------------------
    def track_image(self, frame: np.ndarray, visualize: bool = False, trail_len: int = 30):
        print("[TrackerAPI] track_image: start")
        if frame is None or not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a valid numpy.ndarray")

        print(f"[TrackerAPI] track_image: frame shape={frame.shape}, visualize={visualize}")
        t0 = time.perf_counter()
        frame_res = self.core.track_frame(frame)
        t1 = time.perf_counter()
        print(f"[TrackerAPI] track_image: tracklets={len(frame_res)} time={(t1 - t0):.4f}s")

        if visualize:
            vis = self.visualizer.draw_frame(frame, frame_res, trail_len=int(min(trail_len, 1000)))
            return frame_res, vis
        return frame_res

    # ----------------------------
    # 비디오 파일 추적 + 저장 (+ per-frame on_detection 콜백)
    # ----------------------------
    def track_video(
        self,
        video_path: str,
        save_path: Optional[str] = None,
        trail_len: int = 30,
        *,
        on_detection: Optional[
            # (frame_id, boxes_xyxy, ids, classes, meta) -> None
            Callable[[int, np.ndarray, Optional[np.ndarray], Optional[List[str]], Optional[dict]], None]
        ] = None,
    ):
        print("TrackingAPI : tracking video")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path does not exist: {video_path}")

        self.results = []
        self.core.reset_tracker()
        self.core.img_size = None
        self.visualizer.reset()

        # VideoWriter 준비 (fps/size)
        print(f"[TrackerAPI] encode(camera): prepare writer for {save_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if np.isnan(fps) or fps <= 1e-3:
            fps = 30.0

        if self.decode_target_size is not None:
            writer_w, writer_h = self.decode_target_size
        else:
            width, height = int(cap.get(3)), int(cap.get(4))
            if width == 0 or height == 0:
                ok, f0 = cap.read()
                if not ok or f0 is None:
                    cap.release()
                    raise RuntimeError("Unable to determine video frame size")
                height, width = f0.shape[:2]
                del f0
            writer_w, writer_h = width, height
        cap.release()

        writer, save_path = self._create_writer(save_path, fps, writer_w, writer_h)
        print(f"[TrackerAPI] encode(camera): writer ready path={save_path}, fps={fps:.2f}, size=({writer_w}x{writer_h})")

        # 스트리밍 → 배치 추적 → (옵션: on_detection) → 그리기 → 저장
        frame_count = 0
        print(f"[TrackerAPI] decode(streaming): start "
              f"(threads={self.decode_threads}, prefetch={self.prefetch_frames}, "
              f"hwaccel={self.hwaccel}, target_size={self.decode_target_size})")
        t_all0 = time.perf_counter()

        stream = self.streamer(video_path)

        batch_frames: List[np.ndarray] = []
        batch_raw: List[np.ndarray] = []
        batch_fids: List[int] = []
        last_infer_t = time.perf_counter()

        try:
            print("[TrackerAPI] detect+track+encode(camera): start")
            t_proc0 = time.perf_counter()
            for fid, frame in stream:
                batch_frames.append(frame)
                batch_raw.append(frame)
                batch_fids.append(fid)

                now = time.perf_counter()
                do_flush = False
                if len(batch_frames) >= self.batch_size:
                    do_flush = True
                elif self.infer_timeout > 0 and (now - last_infer_t) >= self.infer_timeout and len(batch_frames) >= self.min_flush:
                    do_flush = True

                if do_flush:
                    t0 = time.perf_counter()
                    batch_res = self.core.track_video_batch(batch_frames)
                    t1 = time.perf_counter()

                    # ── 프레임별 콜백: 디텍션/트래킹 결과가 나온 즉시 전달
                    if on_detection is not None:
                        for _fid, _res in zip(batch_fids, batch_res):
                            boxes_xyxy, ids_array, class_list = self._build_det_arrays(_res)
                            # meta는 필요 시 확장 (타임스탬프가 스트리머에서 오면 여기에 넣기)
                            on_detection(_fid, boxes_xyxy, ids_array, class_list, None)

                    # ── 카메라 비디오 시각화/저장
                    for f, r in zip(batch_raw, batch_res):
                        vis = self.visualizer.draw_frame(f, r, trail_len=int(min(trail_len, 1000)))
                        writer.write(vis)
                        self.results.append(r)
                        frame_count += 1
                        del vis

                    if frame_count % max(1, (self.batch_size * 10)) == 0:
                        print(f"[TrackerAPI] encode(camera): written {frame_count} frames "
                              f"(batch={len(batch_res)}, infer={(t1 - t0):.3f}s)")

                    last_infer_t = now
                    batch_frames.clear()
                    batch_raw.clear()
                    batch_fids.clear()
                    del batch_res
                    gc.collect()

            if batch_frames:
                t0 = time.perf_counter()
                batch_res = self.core.track_video_batch(batch_frames)
                t1 = time.perf_counter()

                if on_detection is not None:
                    for _fid, _res in zip(batch_fids, batch_res):
                        boxes_xyxy, ids_array, class_list = self._build_det_arrays(_res)
                        on_detection(_fid, boxes_xyxy, ids_array, class_list, None)

                for f, r in zip(batch_raw, batch_res):
                    vis = self.visualizer.draw_frame(f, r, trail_len=int(min(trail_len, 1000)))
                    writer.write(vis)
                    self.results.append(r)
                    frame_count += 1
                    del vis

                batch_frames.clear()
                batch_raw.clear()
                batch_fids.clear()
                del batch_res
                gc.collect()
            t_proc1 = time.perf_counter()
        finally:
            writer.release()
            print(f"[TrackerAPI] encode(camera): writer released, frames written: {frame_count}")

        if "t_proc1" not in locals():
            t_proc1 = time.perf_counter()
        t_all1 = time.perf_counter()
        print(f"[INFO] Tracking video saved: {save_path}, frames written: {frame_count}, "
              f"time={(t_all1 - t_all0):.3f}s, proc={(t_proc1 - t_proc0):.3f}s")

        if not self.results:
            raise RuntimeError(f"No frames were processed from: {video_path}")
        return self._pack_results(self.results)

    # ----------------------------
    # 내부: VideoWriter 생성
    # ----------------------------
    def _create_writer(self, save_path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"[TrackerAPI] _create_writer: mp4v OK → {save_path}")
            return writer, save_path

        alt_path = os.path.splitext(save_path)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(alt_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"[TrackerAPI] _create_writer: mp4v fail, MJPG OK → {alt_path}")
            return writer, alt_path

        raise RuntimeError("Failed to open VideoWriter with both mp4v and MJPG")
