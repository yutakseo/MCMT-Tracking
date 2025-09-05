#/workspace/MCMT_engine/stream_SCST.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Any, Callable
import time
import cv2
import numpy as np

from __Detection.detection_api import DetectionAPI
from __Tracking.tracking_api import TrackerAPI
from VideoStreamer.streamer_api import CCTVStreamer
from tools.homo_graphy import PlanProjector


# ----------------------------
# (선택) 추적기 설정값 컨테이너
# ----------------------------
@dataclass
class Args:
    track_thresh = 0.1
    match_thresh = 0.9
    track_buffer = 120
    mot20 = False
    cpu_workers = 20
    chunk_sec = 10.0
    batch_size = 20

# ----------------------------
# Camera pipeline
# ----------------------------
class streamSCST:
    def __init__(
        self,
        cctv_url: str,
        cctv_benchmark: List[Tuple[float, float]],
        plan_path: str,
        plan_benchmark: List[Tuple[float, float]],
        tracker_args: Any,
    ):
        # args
        if tracker_args is None:
            self.traker_args = Args()
        else:
            self.traker_args = tracker_args

        self.cctv_url = cctv_url
        self.cctv_pts = cctv_benchmark
        self.plan = plan_path
        self.plan_pts = plan_benchmark

        # dependencies
        self.detector = DetectionAPI()
        self.tracker = TrackerAPI(args=self.traker_args, detector=self.detector)

        # streamer & warm-up
        self.camera = CCTVStreamer(url=self.cctv_url, max_width=640).start()
        self.camera.wait_ready(timeout=5.0)

        # homography
        self.projector = PlanProjector(
            plan_img_or_path=self.plan,
            trail_len=60,
            trail_ttl=30,
            line_thickness=4,
            point_radius=10, 
        )
        self.H, self.mask = self.projector.fit_homography(
            image_pts=self.cctv_pts, plan_pts=self.plan_pts
        )

        # temp fields
        self.frame: Optional[np.ndarray] = None
        self.detect_result = None
        self.tracklets = None
        self.projectail = None
        self.results = None

    # ---------- internal utils ----------
    @staticmethod
    def _is_valid_frame(f: Any) -> bool:
        return isinstance(f, np.ndarray) and f.size > 0 and f.ndim in (2, 3)

    def _ensure_frame(self, f: Any) -> np.ndarray:
        if not self._is_valid_frame(f):
            raise TypeError("Input frame must be a valid numpy.ndarray")
        return f

    # ---------- stages ----------
    def _videoCapture(self) -> np.ndarray:
        """
        Capture one frame from streamer and store as self.frame.
        """
        frame = self.camera.capture(copy=False)
        # CCTVStreamer가 이미 ndarray를 반환한다면 그대로, 아니라면 np.array로 캐스팅
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame) if frame is not None else None
        self.frame = self._ensure_frame(frame)
        return self.frame

    def _calibration(self, cctv_benchmark, plan_benchmark):
        self.cctv_pts = cctv_benchmark
        self.plan_pts = plan_benchmark
        _, mask = self.projector.fit_homography(
            image_pts=self.cctv_pts, plan_pts=self.plan_pts
        )
        return mask

    def _detection(self, frame: np.ndarray):
        frame = self._ensure_frame(frame)
        self.detect_result = self.detector.detect(frame)
        return self.detect_result

    def _tracking(self, frame: np.ndarray):
        """
        NOTE: 트래커는 '프레임'을 요구함. (det_result는 내부 detector 참조)
        """
        frame = self._ensure_frame(frame)
        self.tracklets = self.tracker.track_image(frame=frame, visualize=False)
        return self.tracklets

    def _projection(self, tracklets):
        self.projectail, self.results = self.projector.projection(
            dets_frame=tracklets, mode="bottom-center", draw=True
        )
        return self.projectail, self.results

    # ---------- public APIs ----------
    def run(self):
        """
        동기 파이프라인: capture -> detect -> track -> project
        """
        frame = self._videoCapture()
        _ = self._detection(frame)            # detector를 쓰는 경우 유지
        tracklets = self._tracking(frame)     # !!! 핵심 수정: 프레임을 넘긴다
        projectail, results = self._projection(tracklets)
        return projectail, results

    def inference(self, frame: Optional[np.ndarray] = None):
        """
        비동기 파이프라인에서 호출:
        - frame이 주어지면 그 프레임으로 추론
        - 없으면 self.frame 사용, 그래도 없으면 캡처해서 사용
        """
        if frame is None:
            if self._is_valid_frame(self.frame):
                frame = self.frame
            else:
                frame = self._videoCapture()

        _ = self._detection(frame)            # detector 사용 유지
        tracklets = self._tracking(frame)     # !!! 핵심 수정: 프레임을 넘긴다
        projectail, results = self._projection(tracklets)
        return projectail, results

    # 선택: 명시적 단일-스텝 API (비동기에서 쓰기 좋음)
    def capture_once(self) -> np.ndarray:
        return self._videoCapture()

    def inference_once(self, frame: np.ndarray):
        frame = self._ensure_frame(frame)
        _ = self._detection(frame)
        tracklets = self._tracking(frame)
        return self._projection(tracklets)

# Backward compatibility: allow `from MCMT_engine.stream_SCST import SCST`
SCST = streamSCST
