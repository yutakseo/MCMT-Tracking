#/workspace/MCMT_engine/core/homoGraphy_renderer.py
import os
import cv2
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

# projected_items 스키마 (PlanProjector.projection 결과):
# { "id": Optional[int], "cls": Optional[str], "bbox": [x1,y1,x2,y2], "pt": (X,Y) }

class PlanRenderer:
    """
    도면 위에 투영된 점/라벨을 그리는 렌더러(계산 분리).
    """

    def __init__(
        self,
        plan_img_or_path: Union[str, np.ndarray],
        color_fn: Optional[Callable[[Dict], Tuple[int, int, int]]] = None,
        point_radius: int = 20,
        line_thickness: int = 2,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_margin: int = 2,
        draw_trails: bool = False,
        trail_len: int = 60,
        trail_ttl: Optional[int] = 30,
    ) -> None:
        if isinstance(plan_img_or_path, str):
            plan = cv2.imread(plan_img_or_path)
            if plan is None:
                raise FileNotFoundError(f"Cannot read plan image: {plan_img_or_path}")
            self.plan = plan
            self.plan_path = plan_img_or_path
        else:
            self.plan = plan_img_or_path.copy()
            self.plan_path = None

        self.color_fn = color_fn or (lambda d: (0, 0, 255))
        self.point_radius = int(point_radius)
        self.line_thickness = int(line_thickness)
        self.text_scale = float(text_scale)
        self.text_thickness = int(text_thickness)
        self.text_margin = int(text_margin)

        # trail 관리(선택)
        self.draw_trails = bool(draw_trails)
        self.trail_len = max(0, int(trail_len))
        self.trail_ttl = int(trail_ttl) if trail_ttl is not None else None
        self._trails = {}  # id -> deque[{"pt":(x,y), "age":age}]
        if self.draw_trails and self.trail_len > 0:
            from collections import defaultdict, deque
            self._trails = defaultdict(lambda: deque(maxlen=self.trail_len))

    # ---------- internal ----------
    def _draw_label(self, canvas: np.ndarray, text: str, org: Tuple[int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, self.text_scale, self.text_thickness)
        x, y = org
        # 배경 박스
        x0 = max(0, x)
        y0 = max(0, y - th - self.text_margin * 2)
        x1 = min(canvas.shape[1] - 1, x + tw + self.text_margin * 2)
        y1 = min(canvas.shape[0] - 1, y + baseline + self.text_margin)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
        # 텍스트
        cv2.putText(
            canvas, text, (x + self.text_margin, y - self.text_margin),
            font, self.text_scale, (255, 255, 255), self.text_thickness, lineType=cv2.LINE_AA
        )

    def _age_and_prune_trails(self, current_ids: Optional[set] = None) -> None:
        if not self.draw_trails or self.trail_len <= 0:
            return
        to_delete = []
        for tid, dq in list(self._trails.items()):
            # age up
            for i in range(len(dq)):
                dq[i]["age"] += 1
            # TTL
            if self.trail_ttl is not None:
                while dq and dq[0]["age"] >= self.trail_ttl:
                    dq.popleft()
            if not dq and (current_ids is None or tid not in current_ids):
                to_delete.append(tid)
        for tid in to_delete:
            del self._trails[tid]

    # ---------- public ----------
    def render_frame(self, projected_items: List[Dict]) -> np.ndarray:
        """
        투영된 항목들을 도면 위에 그려서 canvas 반환.
        projected_items: PlanProjector.projection() 결과
        """
        canvas = self.plan.copy()
        current_ids = set(int(d["id"]) for d in projected_items if d.get("id") is not None)
        self._age_and_prune_trails(current_ids=current_ids)

        for d in projected_items:
            col = self.color_fn(d)
            x, y = map(int, d["pt"])
            # 점
            cv2.circle(canvas, (x, y), self.point_radius, col, -1)
            # 라벨: "ID:7 person" 형태
            parts = []
            if d.get("id") is not None:
                parts.append(f"ID:{int(d['id'])}")
            if d.get("cls") is not None:
                parts.append(str(d["cls"]))
            if parts:
                self._draw_label(canvas, " ".join(parts), (x + 6, y - 6))

            # trail
            if self.draw_trails and self.trail_len > 0 and d.get("id") is not None:
                tid = int(d["id"])
                self._trails[tid].append({"pt": (x, y), "age": 0})

        # trail 그리기
        if self.draw_trails and self.trail_len > 0:
            for tid, dq in self._trails.items():
                pts_alive = [tuple(map(int, p["pt"])) for p in dq
                             if (self.trail_ttl is None or p["age"] < self.trail_ttl)]
                if len(pts_alive) >= 2:
                    col = self.color_fn({"id": tid})
                    pts_np = np.array(pts_alive, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(canvas, [pts_np], False, col, self.line_thickness)
                if len(pts_alive) >= 1:
                    col = self.color_fn({"id": tid})
                    cv2.circle(canvas, pts_alive[-1], self.point_radius, col, -1)

        return canvas

    def save_video(self, projected_frames: List[List[Dict]], out_path: str, fps: float = 30.0) -> None:
        """
        projected_frames: 프레임별 PlanProjector.project_frames(...) 결과
        """
        h, w = self.plan.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError(f"VideoWriter open failed: {out_path}")

        for items in projected_frames:
            canvas = self.render_frame(items)
            vw.write(canvas)

        vw.release()
