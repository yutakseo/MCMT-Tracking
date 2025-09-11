# /workspace/__Tracking/tracking_api.py
import cv2, os, gc, time
import numpy as np
from typing import List, Dict, Any, Iterable, Tuple, Optional, Mapping
from __Tracking.utils.video_decoder import iter_frames_streaming
from __Tracking.core.tracker_core import TrackerCore
from __Tracking.utils.visualizer import TrackerVisualizer


class Args:
    # 트래킹 파라미터
    track_thresh = 0.1
    match_thresh = 0.9
    track_buffer = 180
    mot20 = False

    # 디코더/배치 관련
    # decode_threads 또는 cpu_workers 둘 중 아무거나 제공해도 동작
    cpu_workers = 10               # (구) 이름, 여전히 허용
    decode_threads = None          # (신) 이름, None이면 cpu_workers 사용
    prefetch_frames = 512          # 디코더 프리패치 큐 크기
    hwaccel = None                 # 'cuda' 또는 None
    decode_target_size = None      # (width, height) 지정 시 디코더에서 다운스케일

    batch_size = 20                # GPU 추론 배치 크기

    # 이전 호환(사용 안함, 남겨둠)
    chunk_sec = 10.0


class TrackerAPI:
    def __init__(self, args=None, detector=None) -> None:
        if args is None:
            self.args = Args()
        else:
            self.args = args
        print("TrackingAPI activated...")

        if detector is None or not hasattr(detector, "detect"):
            raise ValueError("detector must be provided and implement a .detect(frame) method")

        self.core = TrackerCore(self.args, detector)

        # decode_threads 우선, 없으면 cpu_workers 사용
        self.decode_threads = int(
            getattr(self.args, "decode_threads",
                    getattr(self.args, "cpu_workers", 8)) or 0
        )
        self.prefetch_frames = int(getattr(self.args, "prefetch_frames", 256))
        self.hwaccel = getattr(self.args, "hwaccel", None)
        self.decode_target_size: Optional[Tuple[int, int]] = getattr(self.args, "decode_target_size", None)

        self.visualizer = TrackerVisualizer()
        self.results: List[List[Dict[str, Any]]] = []
        self.batch_size = int(getattr(self.args, "batch_size", 8))

        print(
            f"[TrackerAPI] __init__: "
            f"decode_threads={self.decode_threads}, prefetch={self.prefetch_frames}, "
            f"hwaccel={self.hwaccel}, target_size={self.decode_target_size}, "
            f"batch_size={self.batch_size}"
        )

    @staticmethod
    def _pack_results(results: List[List[Dict[str, Any]]]) -> Dict[int, List[Dict[str, Any]]]:
        """프레임 인덱스 → 결과 리스트 dict로 포장 (0..N-1 순번)"""
        return {i: frame_res for i, frame_res in enumerate(results)}

    # ----------------------------
    # 내부: 비디오 전체 추적 (저장 X)  → 배치 추론 적용
    # ----------------------------
    def _track_video(self, video_path: str) -> Dict[int, List[Dict[str, Any]]]:
        """비디오 읽어서 추적 결과를 프레임 번호 매핑(dict)로 반환"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path does not exist: {video_path}")

        self.results = []
        self.core.reset_tracker()
        self.core.img_size = None
        self.visualizer.reset()  # 궤적 초기화

        print(
            f"[TrackerAPI] decode(streaming): start "
            f"(threads={self.decode_threads}, prefetch={self.prefetch_frames}, "
            f"hwaccel={self.hwaccel}, target_size={self.decode_target_size})"
        )

        # 순차·순서보장 스트리밍
        stream = iter_frames_streaming(
            video_path,
            target_size=self.decode_target_size,
            decode_threads=self.decode_threads,
            prefetch_frames=self.prefetch_frames,
            hwaccel=self.hwaccel,
            log_prefix="[decoder]",
        )

        batch_frames: List[np.ndarray] = []
        frame_in = 0
        print("[TrackerAPI] detect+track: start")
        t_track0 = time.perf_counter()

        # 디코딩이 느릴 때 GPU가 놀지 않도록 타임아웃 기반 플러시,
        # 단, 너무 작은 배치로 자주 나가지 않게 최소 플러시 크기(min_flush) 도입
        infer_timeout = 0.20  # 0.1~0.3 권장
        min_flush = max(2, self.batch_size // 2)  # 최소 절반 배치 이상 모이도록
        last_infer_t = time.perf_counter()

        for _, frame in stream:
            batch_frames.append(frame)
            now = time.perf_counter()

            do_flush = False
            if len(batch_frames) >= self.batch_size:
                do_flush = True
            elif (now - last_infer_t) >= infer_timeout and len(batch_frames) >= min_flush:
                do_flush = True

            if do_flush:
                t0 = time.perf_counter()
                batch_res = self.core.track_video_batch(batch_frames)  # 디텍션+트래킹
                t1 = time.perf_counter()

                self.results.extend(batch_res)
                frame_in += len(batch_frames)
                last_infer_t = now

                if frame_in % max(1, (self.batch_size * 10)) == 0:
                    print(
                        f"[TrackerAPI] detect+track: processed {frame_in} frames "
                        f"(batch={len(batch_frames)}, infer={(t1 - t0):.3f}s)"
                    )

                batch_frames.clear()
                del batch_res
                gc.collect()

        # 남은 프레임 처리 (배치가 min_flush 이하라도 마지막은 강제 플러시)
        if batch_frames:
            t0 = time.perf_counter()
            batch_res = self.core.track_video_batch(batch_frames)
            t1 = time.perf_counter()
            self.results.extend(batch_res)
            frame_in += len(batch_frames)
            print(
                f"[TrackerAPI] detect+track: processed {frame_in} frames "
                f"(final, infer={(t1 - t0):.3f}s)"
            )
            batch_frames.clear()
            del batch_res
            gc.collect()

        t_track1 = time.perf_counter()
        print(f"[TrackerAPI] detect+track: done frames={len(self.results)} time={(t_track1 - t_track0):.3f}s")

        if not self.results:
            raise RuntimeError(f"No frames were processed from: {video_path}")

        return self._pack_results(self.results)

    # ----------------------------
    # 단일 프레임 추적 (스트리밍 입력용)
    # ----------------------------
    """
    결과값 예시(frame 단일):
    frame_res = [
        {"id": 7,  "bbox": [120.0, 250.0, 300.0, 480.0], "score": 0.92, "class_id": 0,  "label": "worker"},
        {"id": 21, "bbox": [420.0, 200.0, 640.0, 400.0], "score": 0.87, "class_id": 11, "label": "dump_truck"}
    ]
    visualize=True인 경우: (frame_res, vis_img_ndarray)
    """
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
    # 비디오 파일 추적 + 결과 저장 (스트리밍 디코딩 + 배치 추론 한 번 읽기)
    # ----------------------------
    def track_video(self, video_path: str, save_path: str, trail_len: int = 30) -> Dict[int, List[Dict[str, Any]]]:
        print("TrackingAPI : tracking video")
        """비디오 한 번만 읽어서(스트리밍 디코딩) 배치 추적 + 시각화 저장 → 프레임 번호 매핑 반환"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path does not exist: {video_path}")

        # 초기화
        self.results = []
        self.core.reset_tracker()
        self.core.img_size = None
        self.visualizer.reset()  # 궤적 초기화

        # 1) 메타데이터만 확인해서 VideoWriter 준비 (fps)
        print(f"[TrackerAPI] encode(camera): prepare writer for {save_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1e-3:
            fps = 30.0  # fallback

        # writer의 (width, height)는 디코더 출력 크기와 일치해야 함
        if self.decode_target_size is not None:
            writer_w, writer_h = self.decode_target_size
        else:
            width, height = int(cap.get(3)), int(cap.get(4))
            if width == 0 or height == 0:
                ret, frame0 = cap.read()
                if not ret or frame0 is None:
                    cap.release()
                    raise RuntimeError("Unable to determine video frame size")
                height, width = frame0.shape[:2]
                del frame0
            writer_w, writer_h = width, height
        cap.release()

        writer, save_path = self._create_writer(save_path, fps, writer_w, writer_h)
        print(
            f"[TrackerAPI] encode(camera): writer ready path={save_path}, "
            f"fps={fps:.2f}, size=({writer_w}x{writer_h})"
        )

        # 2) 스트리밍 디코더로 프레임 공급 → 배치 추적 + 그리기 + 저장
        frame_count = 0
        print(
            f"[TrackerAPI] decode(streaming): start "
            f"(threads={self.decode_threads}, prefetch={self.prefetch_frames}, "
            f"hwaccel={self.hwaccel}, target_size={self.decode_target_size})"
        )
        t_all0 = time.perf_counter()

        stream = iter_frames_streaming(
            video_path,
            target_size=self.decode_target_size,
            decode_threads=self.decode_threads,
            prefetch_frames=self.prefetch_frames,
            hwaccel=self.hwaccel,
            log_prefix="[decoder]",
        )

        batch_frames: List[np.ndarray] = []
        batch_raw: List[np.ndarray] = []

        # 디코딩 느릴 때 GPU 놀지 않도록 + 너무 잦은 싱글 배치 방지
        infer_timeout = 0.20
        min_flush = max(2, self.batch_size // 2)
        last_infer_t = time.perf_counter()

        try:
            print("[TrackerAPI] detect+track+encode(camera): start")
            t_proc0 = time.perf_counter()
            for _, frame in stream:
                batch_frames.append(frame)
                batch_raw.append(frame)

                now = time.perf_counter()
                do_flush = False
                if len(batch_frames) >= self.batch_size:
                    do_flush = True
                elif (now - last_infer_t) >= infer_timeout and len(batch_frames) >= min_flush:
                    do_flush = True

                if do_flush:
                    t0 = time.perf_counter()
                    batch_res = self.core.track_video_batch(batch_frames)
                    t1 = time.perf_counter()

                    for f, r in zip(batch_raw, batch_res):
                        vis = self.visualizer.draw_frame(
                            f, r, trail_len=int(min(trail_len, 1000))
                        )
                        writer.write(vis)
                        self.results.append(r)
                        frame_count += 1
                        if frame_count % max(1, (self.batch_size * 10)) == 0:
                            print(
                                f"[TrackerAPI] encode(camera): written {frame_count} frames "
                                f"(batch={len(batch_res)}, infer={(t1 - t0):.3f}s)"
                            )
                        del vis

                    last_infer_t = now
                    batch_frames.clear()
                    batch_raw.clear()
                    del batch_res
                    gc.collect()

            # 남은 프레임 처리
            if batch_frames:
                t0 = time.perf_counter()
                batch_res = self.core.track_video_batch(batch_frames)
                t1 = time.perf_counter()
                for f, r in zip(batch_raw, batch_res):
                    vis = self.visualizer.draw_frame(
                        f, r, trail_len=int(min(trail_len, 1000))
                    )
                    writer.write(vis)
                    self.results.append(r)
                    frame_count += 1
                    del vis
                batch_frames.clear()
                batch_raw.clear()
                del batch_res
                gc.collect()
            t_proc1 = time.perf_counter()
        finally:
            writer.release()
            print(f"[TrackerAPI] encode(camera): writer released, frames written: {frame_count}")

        if "t_proc1" not in locals():
            t_proc1 = time.perf_counter()
        t_all1 = time.perf_counter()
        print(
            f"[INFO] Tracking video saved: {save_path}, frames written: {frame_count}, "
            f"time={(t_all1 - t_all0):.3f}s, proc={(t_proc1 - t_proc0):.3f}s"
        )

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
