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
    track_thresh: float = 0.5
    match_thresh: float = 0.5
    track_buffer: int = 60
    mot20: bool = False
    cpu_workers: int = 10


# ----------------------------
# Camera pipeline
# ----------------------------
class SCST:
    def __init__(
        self,
        cctv_url: str,
        cctv_benchmark: List[Tuple[float, float]],
        plan_path: str,
        plan_benchmark: List[Tuple[float, float]],
        tracker_args: Any ,
    ):
    
        if tracker_args is None:    
            self.traker_args = Args()
        else:
            self.traker_args = tracker_args
        
        self.cctv_url = cctv_url
        self.cctv_pts = cctv_benchmark
        self.plan     = plan_path
        self.plan_pts = plan_benchmark

        #initialize dependent module
        self.detector = DetectionAPI()
        self.tracker = TrackerAPI(args=self.traker_args, detector=self.detector)
        self.camera = CCTVStreamer(url=self.cctv_url, max_width=640).start()
        self.camera.wait_ready(timeout=5.0)
        self.projector = PlanProjector(plan_img_or_path=self.plan, trail_len=60, trail_ttl=30, line_thickness=4, point_radius=10)
        self.H, self.mask = self.projector.fit_homography(image_pts=self.cctv_pts, plan_pts=self.plan_pts)
        
        #temp field 
        self.frame:np.ndarray
        self.detect_result= None
        self.tracklets = None
        self.results = None
    
    def _videoCapture(self):
        self.frame = np.array(self.camera.capture(copy=False))
        return self.frame
    
    def _calibration(self, cctv_benchmark, plan_benchmark):
        self.cctv_pts = cctv_benchmark
        self.plan_pts = plan_benchmark
        _, mask = self.projector.fit_homography(image_pts=self.cctv_pts, plan_pts=self.plan_pts)
        return mask

    def _detection(self, frame):
        self.detect_result = self.detector.detect(frame)
        return self.detect_result
    
    def _tracking(self, frame):
        self.tracklets = self.tracker.track_image(frame=frame, visualize=False)
        return self.tracklets
        
    def _projection(self, tracklets):
        self.projectail, self.results = self.projector.projection(dets_frame=tracklets, mode="bottom-center", draw=True)
        return self.projectail, self.results
    
    def run(self):
        frame = self._videoCapture()
        det_result = self._detection(frame)
        tracklets = self._tracking(frame)
        projectail, results = self._projection(tracklets)
        return projectail, results
    
    