# /workspace/MCMT_engine/visualization/visualizer_track.py
import cv2
import numpy as np
from collections import deque
from hashlib import md5
from typing import List, Dict, Any, Optional

class TrackerVisualizer:
    def __init__(self):
        self._trails = {}           # tid -> deque
        self._last_seen = {}        # tid -> last seen frame index (선택)
        self._frame_idx = 0

    def reset(self):
        """저장된 궤적/상태 초기화 (새 영상 시작 시 호출)"""
        self._trails.clear()
        self._last_seen.clear()
        self._frame_idx = 0

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
        trail_ttl: Optional[int] = None,  # 프레임 기준 TTL (예: 300). None이면 현재 프레임에서 사라진 ID는 즉시 제거
    ) -> np.ndarray:
        """
        boxes/ID/클래스/점수/궤적 그리기
        """
        self._frame_idx += 1
        vis = frame.copy() if copy else frame
        current_ids = set()

        for t in frame_res:
            x, y, bw, bh = map(int, t["bbox"])
            tid = int(t["id"])
            score = float(t.get("score", 0.0))
            cls_id = t.get("class_id", None)
            cls_name = t.get("label", None)
            current_ids.add(tid)
            self._last_seen[tid] = self._frame_idx

            color = self._color_from_id(tid)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)

            # 라벨 문자열 구성
            label = f"ID:{tid}"
            if cls_name:
                label += f" {cls_name}"
            elif cls_id is not None:
                label += f" cls:{cls_id}"
            if draw_score:
                label += f" {score:.2f}"

            # 텍스트 출력
            cv2.putText(vis, label, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 궤적 기록
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

        # === 메모리/누수 방지: 현재 프레임에 보이지 않는 ID 정리 ===
        if trail_ttl is None:
            # 즉시 제거 모드: 이번 프레임에 등장하지 않은 ID는 trail 제거
            stale_ids = [tid for tid in list(self._trails.keys()) if tid not in current_ids]
            for tid in stale_ids:
                self._trails.pop(tid, None)
                self._last_seen.pop(tid, None)
        else:
            # TTL 모드: 마지막으로 본 지 trail_ttl 프레임이 지나면 제거
            cutoff = self._frame_idx - int(trail_ttl)
            stale_ids = [tid for tid, last in list(self._last_seen.items()) if last <= cutoff]
            for tid in stale_ids:
                self._trails.pop(tid, None)
                self._last_seen.pop(tid, None)

        return vis
