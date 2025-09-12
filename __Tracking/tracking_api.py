# /workspace/__Tracking/api/tracker_api.py
from typing import List, Dict, Any, Optional
import numpy as np
from __Tracking.engine.tracker_core import TrackerCore

class DefaultArgs:
    track_thresh: float = 0.2
    match_thresh: float = 0.9
    track_buffer: int = 150
    mot20: bool = False

class TrackerAPI:
    def __init__(self, args:Optional[DefaultArgs], detector) -> None:
        if args is None:
            self.args = DefaultArgs()
        else:
            self.args = args

        if detector is None or not hasattr(detector, "detect"):
            raise ValueError("detector.detect(frame) 필요")

        #Tracker Core 인스턴스 생성
        self.core = TrackerCore(self.args, detector)

    def reset(self) -> None:
        self.core.reset_tracker()
        self.core.img_size = None

    def track_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        return self.core.track_frame(frame)

    def track_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        return self.core.track_video_batch(frames)
