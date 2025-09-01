# /workspace/__Tracking/core/tracker_core.py
from __Tracking.ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from typing import List, Dict, Any, Tuple, Optional

class TrackerCore:
    def __init__(self, args, detector) -> None:
        self.args = args
        self.tracker = BYTETracker(args)
        self.detector = detector
        self.img_size: Optional[Tuple[int, int]] = None

    def track_frame(self, frame) -> List[Dict[str, Any]]:
        if frame is None:
            return []
        fh, fw = frame.shape[:2]
        if self.img_size is None:
            self.img_size = (fh, fw)
        dets = self.detector.detect(frame)
        online_targets = self.tracker.update(dets, (fh, fw), self.img_size)
        return [
            {"id": int(t.track_id), "bbox": t.tlwh, "score": float(t.score)}
            for t in online_targets
        ]
