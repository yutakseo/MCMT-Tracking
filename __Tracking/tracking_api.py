import cv2, os
import numpy as np
from typing import List, Dict, Any
from __Tracking.utils.video_decoder import iter_frames_parallel
from __Tracking.core.tracker_core import TrackerCore
from __Tracking.utils.visualizer import TrackerVisualizer


class Args:
    track_thresh = 0.5
    match_thresh = 0.5
    track_buffer = 60
    mot20 = False
    cpu_workers = 10
    chunk_sec   = 10.0


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
        self.chunk_sec   = float(getattr(self.args, "chunk_sec", 20.0))
        self.visualizer = TrackerVisualizer()
        self.results: List[List[Dict[str, Any]]] = []

    # ----------------------------
    # 내부: 비디오 전체 추적 (저장 X)
    # ----------------------------
    def _track_video(self, video_path: str) -> List[List[Dict[str, Any]]]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path does not exist: {video_path}")

        self.results = []
        self.core.img_size = None
        frame_stream = iter_frames_parallel(video_path,
                                            cpu_workers=self.cpu_workers,
                                            chunk_sec=self.chunk_sec)
        for _, frame in frame_stream:
            if frame is None or frame.size == 0:
                continue
            frame_res = self.core.track_frame(frame)
            self.results.append(frame_res)
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
    # 비디오 파일 추적 + 결과 저장
    # ----------------------------
    def track_video(self, video_path: str, save_path: str, trail_len: int = 30) -> List[List[Dict[str, Any]]]:
        # 결과가 없으면 먼저 track_video 실행
        if not self.results:
            print("[INFO] No cached results. Running _track_video() first...")
            self._track_video(video_path)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1e-3:
            fps = 30.0  # fallback

        width, height = int(cap.get(3)), int(cap.get(4))
        if width == 0 or height == 0:
            # fallback: 첫 프레임에서 shape 확인
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("Unable to determine video frame size")
            height, width = frame.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # VideoWriter 생성 (mp4 → mjpg fallback)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            alt_path = os.path.splitext(save_path)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(alt_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError("Failed to open VideoWriter with both mp4v and MJPG")
            save_path = alt_path

        # 결과와 프레임 동기화
        for frame_res in self.results:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            vis = self.visualizer.draw_frame(frame, frame_res, trail_len=int(min(trail_len, 1000)))
            writer.write(vis)

        cap.release()
        writer.release()
        print(f"[INFO] Tracking video saved: {save_path}")
        return self.results
