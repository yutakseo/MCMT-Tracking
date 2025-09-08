# /workspace/MCMT_engine/stream_SCST_shared.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Any, Callable, Dict
import time
import logging
import cv2
import numpy as np

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
# Camera pipeline (모델 공유 버전)
# ----------------------------
class streamSCST:
    def __init__(
        self,
        cctv_url: str,
        cctv_benchmark: List[Tuple[float, float]],
        plan_path: str,
        plan_benchmark: List[Tuple[float, float]],
        tracker_args: Any,
        detector: Optional[Any] = None,  # 외부에서 주입받는 모델
        tracker: Optional[Any] = None,   # 외부에서 주입받는 트래커
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

        # 외부에서 주입받는 모델들 (공유)
        self.detector = detector
        self.tracker = tracker

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

    def _videoCapture(self) -> Optional[np.ndarray]:
        """비디오 캡처 (재시도 로직 포함)"""
        try:
            if self.camera is None:
                return None
            
            frame = self.camera.capture()
            if self._is_valid_frame(frame):
                self.frame = frame
                return frame
            else:
                return None
                
        except Exception as e:
            logging.error(f"Video capture failed: {e}")
            return None

    def _detection(self, frame: np.ndarray) -> Any:
        """객체 탐지 (공유 모델 사용)"""
        if self.detector is None:
            return []
        
        try:
            timestamp = time.time()
            result = self.detector.detect(frame)
            return result
        except Exception as e:
            logging.error(f"Detection failed: {e}")
            return []

    def _tracking(self, frame: np.ndarray) -> Any:
        """추적 (공유 트래커 사용)"""
        if self.tracker is None:
            print(f"[DEBUG] streamSCST._tracking: tracker is None, 빈 리스트 반환")
            return []
        
        try:
            print(f"[DEBUG] streamSCST._tracking: tracker.track_image 호출 시작...")
            result = self.tracker.track_image(frame)
            print(f"[DEBUG] streamSCST._tracking: tracker.track_image 완료, {len(result)} tracklets")
            print(f"[DEBUG] streamSCST._tracking: result 타입: {type(result)}")
            if result:
                print(f"[DEBUG] streamSCST._tracking: 첫 번째 tracklet 예시: {result[0] if len(result) > 0 else 'None'}")
            return result
        except Exception as e:
            logging.error(f"Tracking failed: {e}")
            print(f"[DEBUG] streamSCST._tracking: 예외 발생: {e}")
            return []

    def _projection(self, tracklets: Any) -> Tuple[List[Dict], Any]:
        """호모그래피 변환"""
        try:
            timestamp = time.time()
            projected, trails = self.projector.projection(tracklets, mode="center", timestamp=timestamp)
            return projected, trails
        except Exception as e:
            logging.error(f"Projection failed: {e}")
            return [], None

    def run(self) -> Dict[str, Any]:
        """단일 프레임 처리"""
        timestamp = time.time()
        
        # 1. 프레임 캡처
        frame = self._videoCapture()
        if frame is None:
            return self._empty_result(timestamp)
        
        # 2. 객체 탐지
        self.detect_result = self._detection(frame)
        
        # 3. 추적
        self.tracklets = self._tracking(frame)
        
        # 4. 호모그래피 변환
        self.projectail, trails = self._projection(self.tracklets)
        
        # 5. 결과 패키징
        return {
            'timestamp': timestamp,
            'frame': frame,
            'detections': self.detect_result,
            'tracklets': self.tracklets,
            'projected': self.projectail,
            'trails': trails
        }

    def inference(self) -> Dict[str, Any]:
        """추론만 실행 (프레임 캡처 제외)"""
        timestamp = time.time()
        
        if self.frame is None:
            return self._empty_result(timestamp)
        
        # 1. 객체 탐지
        self.detect_result = self._detection(self.frame)
        
        # 2. 추적
        self.tracklets = self._tracking(self.frame)
        
        # 3. 호모그래피 변환
        self.projectail, trails = self._projection(self.tracklets)
        
        # 4. 결과 패키징
        return {
            'timestamp': timestamp,
            'frame': self.frame,
            'detections': self.detect_result,
            'tracklets': self.tracklets,
            'projected': self.projectail,
            'trails': trails
        }

    def inference_once(self, frame: np.ndarray) -> Dict[str, Any]:
        """단일 프레임 추론"""
        timestamp = time.time()
        
        # 1. 객체 탐지
        detections = self._detection(frame)
        
        # 2. 추적
        tracklets = self._tracking(frame)
        
        # 3. 호모그래피 변환
        projected, trails = self._projection(tracklets)
        
        # 4. 결과 패키징
        return {
            'timestamp': timestamp,
            'frame': frame,
            'detections': detections,
            'tracklets': tracklets,
            'projected': projected,
            'trails': trails
        }

    def _empty_result(self, timestamp: float) -> Dict[str, Any]:
        """빈 결과 반환"""
        return {
            'timestamp': timestamp,
            'frame': None,
            'detections': [],
            'tracklets': [],
            'projected': [],
            'trails': None
        }

    def close(self):
        """리소스 정리"""
        if hasattr(self, 'camera') and self.camera is not None:
            try:
                self.camera.stop()
            except:
                pass
            self.camera = None
        
        # 모델은 외부에서 관리되므로 여기서는 정리하지 않음
        logging.info("✅ streamSCST 리소스 정리 완료")


# 하위 호환성을 위한 별칭
SCST = streamSCST
