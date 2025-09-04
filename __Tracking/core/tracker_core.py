# /workspace/__Tracking/core/tracker_core.py
from __Tracking.ByteTrack.yolox.tracker.byte_tracker import BYTETracker, STrack
from typing import List, Dict, Any, Tuple, Optional

class TrackerCore:
    def __init__(self, args, detector) -> None:
        self.args = args
        self.tracker = BYTETracker(args)
        self.detector = detector
        self.img_size: Optional[Tuple[int, int]] = None

        # detector에서 class_id → name 맵 추출 (있다면)
        if hasattr(detector, "name_map"):
            self.class_map = detector.name_map()
        else:
            self.class_map = {}

    def track_frame(self, frame) -> List[Dict[str, Any]]:
        if frame is None:
            return []
        fh, fw = frame.shape[:2]
        if self.img_size is None:
            self.img_size = (fh, fw)

        # 1) DetectionAPI → torch.Tensor (N,6)
        dets = self.detector.detect(frame)

        # 2) ByteTrack 업데이트
        online_targets = self.tracker.update(dets, (fh, fw), self.img_size)

        # 3) 결과 변환 (데이터형 변환 최소화)
        results = []
        for t in online_targets:
            class_id = t.class_id if t.class_id is not None else -1
            label = self.class_map.get(class_id, class_id)  # 이름 없으면 class_id 그대로
            results.append({
                "id": t.track_id,      # int 유지
                "class_id": class_id,  # np.int32 그대로 보존
                "label": label,        # str 또는 np.int32
                "bbox": t.tlwh,        # np.ndarray(float32)
                "score": t.score,      # np.float32 그대로
            })
        return results
