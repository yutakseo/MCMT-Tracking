# /workspace/__Tracking/utils/visualizer.py
import cv2
import numpy as np
from collections import deque
from hashlib import md5
from typing import List, Dict, Any, Optional

class TrackerVisualizer:
    def __init__(self):
        self._trails = {}

    def _color_from_id(self, track_id: int):
        h = md5(str(track_id).encode()).hexdigest()
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))  # BGR

    def draw_frame(
        self,
        frame: np.ndarray,
        frame_res: List[Dict[str, Any]],
        trail_len: int = 30,
        trail_thickness: int = 2,
        draw_score: bool = True,
        copy: bool = True,
    ) -> np.ndarray:
        """
        boxes/ID/점수/궤적 그리기
        """
        vis = frame.copy() if copy else frame
        current_ids = set()

        for t in frame_res:
            x, y, bw, bh = map(int, t["bbox"])
            tid = int(t["id"])
            score = float(t.get("score", 0.0))
            current_ids.add(tid)

            color = self._color_from_id(tid)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
            label = f"ID:{tid}" + (f" {score:.2f}" if draw_score else "")
            cv2.putText(vis, label, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if trail_len > 0:
                if tid not in self._trails:
                    self._trails[tid] = deque(maxlen=trail_len)
                cx, cy = x + bw * 0.5, y + bh * 0.5
                self._trails[tid].append((cx, cy))

        # 궤적 그리기
        if trail_len > 0:
            for tid in current_ids:
                pts = self._trails.get(tid, None)
                if pts and len(pts) >= 2:
                    pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [pts_np], isClosed=False,
                                  color=self._color_from_id(tid),
                                  thickness=trail_thickness)
                if pts and len(pts) >= 1:
                    cx, cy = map(int, pts[-1])
                    cv2.circle(vis, (cx, cy), radius=2 + trail_thickness,
                               color=self._color_from_id(tid), thickness=-1)

        return vis
