# /workspace/__Tracking/tracking_api.py
import cv2, os, gc
import numpy as np
from typing import List, Dict, Any, Iterable, Tuple
from __Tracking.utils.video_decoder import iter_frames_parallel
from __Tracking.core.tracker_core import TrackerCore
from __Tracking.utils.visualizer import TrackerVisualizer


class Args:
    track_thresh = 0.1
    match_thresh = 0.9
    track_buffer = 180
    mot20 = False
    cpu_workers = 10
    chunk_sec   = 10.0 
    batch_size = 20


class TrackerAPI:
    def __init__(self, args=None, detector=None) -> None:
        if args is None:
            self.args = Args()
        else:
            self.args = args

        if detector is None or not hasattr(detector, "detect"):
            raise ValueError("detector must be provided and implement a .detect(frame) method")

        self.core = TrackerCore(self.args, detector)
        self.cpu_workers = int(getattr(self.args, "cpu_workers", 8))
        self.chunk_sec = float(getattr(self.args, "chunk_sec", 20.0))
        self.visualizer = TrackerVisualizer()
        self.results: List[List[Dict[str, Any]]] = []
        self.batch_size = int(getattr(self.args, "batch_size", 8))

        # 재정렬 버퍼의 최대 허용 크기(Out-of-order 완충; 너무 커지면 메모리 증가)
        self._max_reorder_buffer = max(100, self.batch_size * 8)

    # ---- 내부 유틸: 병렬 디코딩 결과를 전역 프레임 인덱스 순서로 정렬 ----
    def _ordered_stream(self, frame_stream: Iterable[Tuple[int, np.ndarray]]):
        """
        iter_frames_parallel는 chunk 단위 병렬로 인해 (chunk 순서가) out-of-order가 될 수 있음.
        여기서 전역 프레임 인덱스 기준으로 재정렬해 순차적으로 yield.
        메모리 안전장치: 버퍼가 너무 커지면(희소하게 들어오는 경우) 낮은 인덱스부터 부분 flush.
        """
        buffer = {}          # idx -> frame
        expected = None      # 다음에 내보낼 전역 인덱스
        for idx, frame in frame_stream:
            if frame is None or frame.size == 0:
                continue
            buffer[idx] = frame
            if expected is None:
                expected = idx

            # 가능한 한 연속으로 비우기
            while expected in buffer:
                f = buffer.pop(expected)
                yield expected, f
                expected += 1

            # 메모리 안전장치: 버퍼가 너무 커지면 낮은 인덱스부터 부분 flush
            if len(buffer) > self._max_reorder_buffer:
                # 가장 낮은 쪽부터 expected에 가까운 순서로 정렬하여 일부 방출
                keys_sorted = sorted(buffer.keys())
                # expected보다 앞선(지나간) 프레임은 없겠지만 혹시 모를 이상치 제거
                flush_keys = [k for k in keys_sorted if k < expected]
                for k in flush_keys:
                    f = buffer.pop(k, None)
                    if f is not None:
                        # 순서가 확실치 않으면 그냥 건너뛰는 대신 방출
                        yield k, f

        # 남은 게 있다면 정렬해서 마무리 (이상 케이스)
        if buffer:
            for k in sorted(buffer.keys()):
                yield k, buffer[k]
        buffer.clear()
        gc.collect()

    # ----------------------------
    # 내부: 비디오 전체 추적 (저장 X)  → 배치 추론 적용
    # ----------------------------
    def _track_video(self, video_path: str) -> List[List[Dict[str, Any]]]:
        """비디오 읽어서 추적 결과만 리스트로 반환 (배치 추론/병렬 디코딩 + 순서 재정렬)"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path does not exist: {video_path}")

        self.results = []
        self.core.reset_tracker()
        self.core.img_size = None
        self.visualizer.reset()  # 궤적 초기화

        raw_stream = iter_frames_parallel(
            video_path,
            cpu_workers=self.cpu_workers,
            chunk_sec=self.chunk_sec,
        )
        stream = self._ordered_stream(raw_stream)  # 순서 보장 스트림

        batch_frames: List[np.ndarray] = []
        for _, frame in stream:
            batch_frames.append(frame)
            if len(batch_frames) >= self.batch_size:
                batch_res = self.core.track_video_batch(batch_frames)
                self.results.extend(batch_res)
                # 메모리 해제
                batch_frames.clear()
                del batch_res
                gc.collect()

        # 남은 프레임 처리
        if batch_frames:
            batch_res = self.core.track_video_batch(batch_frames)
            self.results.extend(batch_res)
            batch_frames.clear()
            del batch_res
            gc.collect()

        if not self.results:
            raise RuntimeError(f"No frames were processed from: {video_path}")

        return self.results

    # ----------------------------
    # 단일 프레임 추적 (스트리밍 입력용)
    # ----------------------------
    def track_image(self, frame: np.ndarray, visualize: bool = False, trail_len: int = 30):
        if frame is None or not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a valid numpy.ndarray")

        frame_res = self.core.track_frame(frame)
        if visualize:
            vis = self.visualizer.draw_frame(frame, frame_res, trail_len=int(min(trail_len, 1000)))
            return frame_res, vis
        return frame_res

    # ----------------------------
    # 비디오 파일 추적 + 결과 저장 (병렬 디코딩 + 배치 추론으로 한 번 읽기)
    # ----------------------------
    def track_video(self, video_path: str, save_path: str, trail_len: int = 30) -> List[List[Dict[str, Any]]]:
        """비디오 한 번만 읽어서(병렬 디코딩) 배치 추적 + 시각화 저장 (전역 프레임 순서 보장)"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path does not exist: {video_path}")

        # 초기화
        self.results = []
        self.core.reset_tracker()
        self.core.img_size = None
        self.visualizer.reset()  # 궤적 초기화

        # 1) 메타데이터만 확인해서 VideoWriter 준비 (fps/size)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1e-3:
            fps = 30.0  # fallback
        width, height = int(cap.get(3)), int(cap.get(4))
        if width == 0 or height == 0:
            ret, frame0 = cap.read()
            if not ret or frame0 is None:
                cap.release()
                raise RuntimeError("Unable to determine video frame size")
            height, width = frame0.shape[:2]
            del frame0
        cap.release()

        writer, save_path = self._create_writer(save_path, fps, width, height)

        # 2) 병렬 디코더로 프레임 공급 → (정렬) 배치 추적 + 그리기 + 저장
        frame_count = 0
        raw_stream = iter_frames_parallel(
            video_path,
            cpu_workers=self.cpu_workers,
            chunk_sec=self.chunk_sec,
        )
        stream = self._ordered_stream(raw_stream)  # 순서 보장 스트림

        batch_frames: List[np.ndarray] = []
        batch_raw: List[np.ndarray] = []

        try:
            for _, frame in stream:
                batch_frames.append(frame)
                batch_raw.append(frame)

                if len(batch_frames) >= self.batch_size:
                    batch_res = self.core.track_video_batch(batch_frames)

                    for f, r in zip(batch_raw, batch_res):
                        vis = self.visualizer.draw_frame(
                            f, r, trail_len=int(min(trail_len, 1000))
                        )
                        writer.write(vis)
                        self.results.append(r)
                        frame_count += 1
                        # 프레임/시각화 이미지 참조 해제
                        del vis
                    # 배치 버퍼 비우기 + 가비지 컬렉션
                    batch_frames.clear()
                    batch_raw.clear()
                    del batch_res
                    gc.collect()

            # 남은 프레임 처리
            if batch_frames:
                batch_res = self.core.track_video_batch(batch_frames)
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
        finally:
            writer.release()

        print(f"[INFO] Tracking video saved: {save_path}, frames written: {frame_count}")
        return self.results

    # ----------------------------
    # 내부: VideoWriter 생성
    # ----------------------------
    def _create_writer(self, save_path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            alt_path = os.path.splitext(save_path)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(alt_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Failed to open VideoWriter with both mp4v and MJPG")
            save_path = alt_path
        return writer, save_path
