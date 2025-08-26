# /workspace/__Visualization/plan_projector.py
import os
import cv2
import numpy as np
from collections import defaultdict, deque
from hashlib import md5
from typing import Callable, Dict, List, Optional, Tuple, Union

DetItem = Dict[str, Union[int, float, list, np.ndarray]]

class PlanProjector:
    """
    도면(평면) 위로 객체탐지/트래킹 결과를 투영/시각화하는 유틸 클래스.

    Parameters
    ----------
    plan_img_or_path : np.ndarray | str
        도면 이미지(BGR) 또는 경로.
    H : np.ndarray | None
        이미지 → 도면 호모그래피(3x3, float32). 없으면 fit_homography()나
        projection(image_pts, plan_pts=...) 호출에서 추정 가능.
    color_fn : Callable[[DetItem], Tuple[int,int,int]] | None
        각 항목 색상 함수. None이면 기본 빨강.
    trail_len : int
        궤적 길이(최근 좌표 저장 개수). 0이면 궤적 미사용.
    """

    def __init__(
        self,
        plan_img_or_path: Union[str, np.ndarray],
        H: Optional[np.ndarray] = None,
        color_fn: Optional[Callable[[DetItem], Tuple[int,int,int]]] = None,
        trail_len: int = 0
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

        self.H: Optional[np.ndarray] = None
        if H is not None:
            self.set_homography(H)

        self.color_fn = color_fn
        self.trail_len = max(0, int(trail_len))
        self.trails = defaultdict(lambda: deque(maxlen=self.trail_len)) if self.trail_len > 0 else None

    # -------------------- Homography 유틸 --------------------
    def set_homography(self, H: np.ndarray) -> None:
        H = np.asarray(H, dtype=np.float32)
        if H.shape != (3, 3):
            raise ValueError("H must be a 3x3 matrix")
        self.H = H

    def fit_homography(
        self,
        image_pts: List[Tuple[int, int]],
        plan_pts: List[Tuple[int, int]],
        ransac_thresh: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        (image_pts ↔ plan_pts)으로 H를 계산하고 내부에 설정.

        image_pts : [(x1,y1), (x2,y2), ...]  (영상 좌표)
        plan_pts  : [(X1,Y1), (X2,Y2), ...]  (도면 좌표)
        """
        ip = np.asarray(image_pts, np.float32)
        pp = np.asarray(plan_pts,  np.float32)
        if ip.shape[0] < 4 or pp.shape[0] < 4:
            raise ValueError("Need >= 4 point correspondences.")

        H, mask = cv2.findHomography(ip, pp, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if H is None:
            raise RuntimeError("findHomography failed.")
        self.set_homography(H)
        return H, mask

    def save_h(self, path: str) -> None:
        if self.H is None:
            raise RuntimeError("No homography to save.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path, self.H.astype(np.float32))

    def load_h(self, path: str) -> None:
        H = np.load(path).astype(np.float32)
        self.set_homography(H)

    # -------------------- 내부 유틸 --------------------
    @staticmethod
    def _xyxy_from_any(bbox: Union[list, np.ndarray]) -> np.ndarray:
        """
        bbox가 tlwh([x,y,w,h]) 또는 xyxy([x1,y1,x2,y2])일 수 있으므로 xyxy로 표준화
        """
        b = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if b.shape[0] < 4:
            raise ValueError("bbox length must be >= 4")

        x1, y1 = b[0], b[1]
        # tlwh로 판단: w,h > 0 이고 너무 큰 값이 아닌 경우
        if b[2] > 0 and b[3] > 0 and (b[2] + b[3]) < 1e6:
            w, h = b[2], b[3]
            return np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32)

        # 그 외는 xyxy로 간주 후 정렬
        x2, y2 = b[2], b[3]
        x1, y1, x2, y2 = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    @staticmethod
    def _project_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
        """(N,2) pts_xy를 (3x3) H로 투영하여 (N,2) 반환"""
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, H)
        return proj.reshape(-1, 2)

    @staticmethod
    def _default_color(_: DetItem) -> Tuple[int,int,int]:
        return (0, 0, 255)  # 빨강

    @staticmethod
    def _hash_color(track_id: int) -> Tuple[int,int,int]:
        """ID 고정 색상 (원하면 color_fn에 사용)"""
        h = md5(str(track_id).encode()).hexdigest()
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

    # -------------------- 핵심: 한 프레임 투영/시각화 --------------------
    def projection(
        self,
        dets_frame: List[DetItem],
        mode: str = "center",
        draw: bool = True,
        image_pts: Optional[List[Tuple[float, float]]] = None,
        plan_pts: Optional[List[Tuple[float, float]]] = None,
        ransac_thresh: float = 3.0
    ) -> Tuple[List[DetItem], Optional[np.ndarray]]:
        """
        한 프레임의 탐지/트랙 결과를 도면 좌표계로 투영.

        Parameters
        ----------
        dets_frame : [{'id':tid, 'bbox':tlwh or xyxy, 'score':...}, ...]
        mode : 'center' | 'corners'
            - center: 중심점만 투영 (일반적으로 안정적)
            - corners: 네 꼭짓점 폴리곤 투영
        draw : bool
            True면 도면 위에 그린 캔버스도 함께 반환
        image_pts : [(x1,y1), (x2,y2), ...] | None
            원 영상 좌표 (넘기면 이 호출에서 H를 새로 계산)
        plan_pts  : [(X1,Y1), (X2,Y2), ...] | None
            도면 좌표 (넘기면 이 호출에서 H를 새로 계산)
        ransac_thresh : float
            findHomography RANSAC 임계값

        Returns
        -------
        projected : List[DetItem]
            center → 각 항목에 'pt': (X,Y)
            corners → 각 항목에 'quad': [(X1,Y1),...(X4,Y4)]
        canvas : np.ndarray | None
            draw=True일 때 도면 위 시각화 이미지
        """
        # 1) H 확보
        if image_pts is not None and plan_pts is not None:
            ip = np.asarray(image_pts, np.float32)
            pp = np.asarray(plan_pts,  np.float32)
            H, _ = cv2.findHomography(ip, pp, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
            if H is None:
                raise RuntimeError("findHomography failed inside projection()")
        else:
            if self.H is None:
                raise RuntimeError("Homography (H) not set. Set it or pass image_pts/plan_pts.")
            H = self.H

        # 2) 투영 변환
        projected: List[DetItem] = []
        for d in dets_frame:
            bbox = d.get("bbox", d.get("box", d.get("tlwh", None)))
            if bbox is None:
                continue
            xyxy = self._xyxy_from_any(bbox)
            x1, y1, x2, y2 = xyxy

            item = dict(d)
            if mode == "center":
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                p = self._project_points(H, np.array([[cx, cy]], np.float32))[0]
                item["pt"] = (float(p[0]), float(p[1]))
            else:
                quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
                quad_p = self._project_points(H, quad)
                item["quad"] = [ (float(px), float(py)) for px, py in quad_p ]
            projected.append(item)

            # 궤적 저장(옵션)
            if self.trails is not None and "id" in item:
                tid = int(item["id"])
                if mode == "center":
                    self.trails[tid].append(item["pt"])
                else:
                    # corners인 경우에도 중심점으로 궤적 저장
                    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                    p = self._project_points(H, np.array([[cx, cy]], np.float32))[0]
                    self.trails[tid].append((float(p[0]), float(p[1])))

        # 3) 시각화
        canvas = None
        if draw:
            canvas = self.plan.copy()
            for d in projected:
                col = (self.color_fn or self._default_color)(d)
                if mode == "center":
                    x, y = map(int, d["pt"])
                    cv2.circle(canvas, (x, y), 4, col, -1)
                    if "id" in d:
                        cv2.putText(canvas, f"ID:{int(d['id'])}", (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                else:
                    quad = np.asarray(d["quad"], np.int32).reshape(-1, 1, 2)
                    cv2.polylines(canvas, [quad], False, col, 2)
                    if "id" in d:
                        q0 = tuple(quad[0, 0])
                        cv2.putText(canvas, f"ID:{int(d['id'])}", (q0[0] + 5, q0[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            # 궤적
            if self.trails is not None:
                for tid, pts in self.trails.items():
                    if len(pts) >= 2:
                        pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                        col = (self.color_fn or self._default_color)({"id": tid})
                        cv2.polylines(canvas, [pts_np], False, col, 2)
                    if len(pts) >= 1:
                        cx, cy = map(int, pts[-1])
                        col = (self.color_fn or self._default_color)({"id": tid})
                        cv2.circle(canvas, (cx, cy), 4, col, -1)

        return projected, canvas

    # -------------------- 여러 프레임을 도면 비디오로 저장 --------------------
    def save_video(
        self,
        frames: List[List[DetItem]],
        out_path: str,
        fps: int = 30,
        mode: str = "center",
        image_pts: Optional[List[Tuple[float, float]]] = None,
        plan_pts: Optional[List[Tuple[float, float]]] = None,
        ransac_thresh: float = 3.0
    ) -> None:
        """
        frames(프레임별 dets 리스트)를 투영하여 도면 비디오로 저장.
        필요한 경우 호출마다 image_pts/plan_pts로 H를 계산.
        """
        h, w = self.plan.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        vw = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not vw.isOpened():
            raise RuntimeError(f"VideoWriter open failed: {out_path}")

        for frame_res in frames:
            _, canvas = self.projection(
                frame_res, mode=mode, draw=True,
                image_pts=image_pts, plan_pts=plan_pts, ransac_thresh=ransac_thresh
            )
            vw.write(canvas)

        vw.release()