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
    trail_ttl : int | None
        궤적의 프레임 생존 시간. None이면 무제한( trail_len만 적용 ).
    line_thickness : int
        폴리라인/궤적 선 굵기(px).
    point_radius : int
        점(원) 반지름(px).
    """

    def __init__(
        self,
        plan_img_or_path: Union[str, np.ndarray],
        H: Optional[np.ndarray] = None,
        color_fn: Optional[Callable[[DetItem], Tuple[int,int,int]]] = None,
        trail_len: int = 0,
        trail_ttl: Optional[int] = None,
        line_thickness: int = 2,
        point_radius: int = 4,
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
        self.trail_ttl = int(trail_ttl) if trail_ttl is not None else None
        self.line_thickness = int(line_thickness)
        self.point_radius = int(point_radius)

        # trails: id -> deque of {'pt': (x,y), 'age': int}
        if self.trail_len > 0:
            self.trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.trail_len))
        else:
            self.trails = None

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
        """
        ip = np.asarray(image_pts, np.float32)
        pp = np.asarray(plan_pts,  np.float32)
        if ip.shape[0] < 4 or pp.shape[0] < 4 or ip.shape[0] != pp.shape[0]:
            raise ValueError("Need >= 4 correspondences and the same number of image_pts and plan_pts.")

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
        bbox가 tlwh([x,y,w,h]) 또는 xyxy([x1,y1,x2,y2])일 수 있으므로 xyxy로 표준화.
        우선 xyxy로 그럴듯하면 xyxy, 아니면 tlwh를 xyxy로 변환.
        """
        b = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if b.shape[0] < 4:
            raise ValueError("bbox length must be >= 4")

        x1, y1, a, b2 = b[0], b[1], b[2], b[3]

        # xyxy로 그럴듯한 경우(x2>x1, y2>y1)
        if a > x1 and b2 > y1:
            return np.array([x1, y1, a, b2], dtype=np.float32)

        # tlwh로 그럴듯한 경우(w>0, h>0)
        if a > 0 and b2 > 0:
            return np.array([x1, y1, x1 + a, y1 + b2], dtype=np.float32)

        # 마지막 방어: 정렬해서 xyxy 반환
        x2, y2 = a, b2
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

    # -------------------- Trail aging helper --------------------
    def _age_and_prune_trails(self, current_ids: Optional[set] = None) -> None:
        """모든 trail의 age를 +1 하고, trail_ttl 초과분 제거. 비어지면 id 삭제."""
        if self.trails is None:
            return
        to_delete = []
        for tid, dq in self.trails.items():
            for i in range(len(dq)):
                dq[i]["age"] += 1
            if self.trail_ttl is not None:
                while dq and dq[0]["age"] >= self.trail_ttl:
                    dq.popleft()
            if not dq and (current_ids is None or tid not in current_ids):
                to_delete.append(tid)
        for tid in to_delete:
            del self.trails[tid]

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
        """
        # mode 정규화 (하이픈 표기 허용)
        if mode == "bottom-center":
            mode = "bottom_center"

        # 1) H 확보
        if image_pts is not None and plan_pts is not None:
            ip = np.asarray(image_pts, np.float32)
            pp = np.asarray(plan_pts,  np.float32)
            if ip.shape[0] < 4 or pp.shape[0] < 4 or ip.shape[0] != pp.shape[0]:
                raise ValueError("Need >= 4 correspondences and the same number of image_pts and plan_pts.")
            H, _ = cv2.findHomography(ip, pp, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
            if H is None:
                raise RuntimeError("findHomography failed inside projection()")
        else:
            if self.H is None:
                raise RuntimeError("Homography (H) not set. Set it or pass image_pts/plan_pts.")
            H = self.H

        current_ids = {int(d["id"]) for d in dets_frame if "id" in d}

        # 2-a) Trail aging & TTL
        self._age_and_prune_trails(current_ids=current_ids)

        # 2-b) 투영 변환
        projected: List[DetItem] = []
        for d in dets_frame:
            bbox = d.get("bbox", d.get("box", d.get("tlwh", None)))
            if bbox is None:
                continue
            x1, y1, x2, y2 = self._xyxy_from_any(bbox)

            item = dict(d)
            if mode == "center":
                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                p = self._project_points(H, np.array([[cx, cy]], np.float32))[0]
                item["pt"] = (float(p[0]), float(p[1]))

            elif mode == "bottom_center":
                cx, cy = (x1 + x2) * 0.5, y2
                p = self._project_points(H, np.array([[cx, cy]], np.float32))[0]
                item["pt"] = (float(p[0]), float(p[1]))

            elif mode == "corners":
                quad = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
                quad_p = self._project_points(H, quad)
                item["quad"] = [ (float(px), float(py)) for px, py in quad_p ]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            projected.append(item)

            # 2-c) 궤적 업데이트(현재 프레임 점 추가, age=0)
            if self.trails is not None and "id" in item:
                if mode in ("center", "bottom_center"):
                    tid = int(item["id"])
                    self.trails.setdefault(tid, deque(maxlen=self.trail_len))
                    self.trails[tid].append({"pt": item["pt"], "age": 0})
                elif mode == "corners":
                    # 코너 모드에서는 중심점 기준으로 trail
                    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                    p = self._project_points(H, np.array([[cx, cy]], np.float32))[0]
                    tid = int(item["id"])
                    self.trails.setdefault(tid, deque(maxlen=self.trail_len))
                    self.trails[tid].append({"pt": (float(p[0]), float(p[1])), "age": 0})

        # 3) 시각화
        canvas = None
        if draw:
            canvas = self.plan.copy()
            for d in projected:
                col = (self.color_fn or self._default_color)(d)

                if mode in ("center", "bottom_center"):
                    x, y = map(int, d["pt"])
                    cv2.circle(canvas, (x, y), self.point_radius, col, -1)
                    if "id" in d:
                        cv2.putText(canvas, f"ID:{int(d['id'])}", (x + 5, y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

                elif mode == "corners":
                    quad = np.asarray(d["quad"], np.int32).reshape(-1, 1, 2)
                    cv2.polylines(canvas, [quad], False, col, self.line_thickness)
                    if "id" in d:
                        q0 = tuple(quad[0, 0])
                        cv2.putText(canvas, f"ID:{int(d['id'])}", (q0[0] + 5, q0[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

            # 궤적(살아있는 점만)
            if self.trails is not None:
                for tid, dq in self.trails.items():
                    pts_alive = [tuple(map(int, p["pt"])) for p in dq if (self.trail_ttl is None or p["age"] < self.trail_ttl)]
                    if len(pts_alive) >= 2:
                        col = (self.color_fn or self._default_color)({"id": tid})
                        pts_np = np.array(pts_alive, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(canvas, [pts_np], False, col, self.line_thickness)
                    if len(pts_alive) >= 1:
                        col = (self.color_fn or self._default_color)({"id": tid})
                        cv2.circle(canvas, pts_alive[-1], self.point_radius, col, -1)

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
        ransac_thresh: float = 3.0,
        codec: str = "MJPG",
        out_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        frames(프레임별 dets 리스트)를 투영하여 도면 비디오로 저장.

        codec:
          - Windows 보편 호환: 'MJPG' + .avi
          - H.264(OpenCV 빌드에 따라): 'avc1' 또는 'H264' + .mp4
          - 기본: 'mp4v' + .mp4
        out_size:
          - 지정 시 모든 프레임을 (w,h)로 리사이즈 후 저장
        """
        # mode 정규화 (하이픈 표기 허용)
        if mode == "bottom-center":
            mode = "bottom_center"

        base_h, base_w = self.plan.shape[:2]
        if out_size is None:
            out_w, out_h = base_w, base_h
        else:
            out_w, out_h = int(out_size[0]), int(out_size[1])

        fourcc = cv2.VideoWriter_fourcc(*codec)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        vw = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
        if not vw.isOpened():
            raise RuntimeError(f"VideoWriter open failed: {out_path} (codec={codec})")

        def _prepare_frame(bgr: np.ndarray) -> np.ndarray:
            fr = bgr
            if fr.ndim == 2:
                fr = cv2.cvtColor(fr, cv2.COLOR_GRAY2BGR)
            elif fr.shape[2] == 4:
                fr = fr[:, :, :3]
            if (fr.shape[1], fr.shape[0]) != (out_w, out_h):
                fr = cv2.resize(fr, (out_w, out_h), interpolation=cv2.INTER_AREA)
            if fr.dtype != np.uint8:
                fr = fr.astype(np.uint8, copy=False)
            return np.ascontiguousarray(fr)

        wrote = 0
        for frame_res in frames:
            _, canvas = self.projection(
                frame_res, mode=mode, draw=True,
                image_pts=image_pts, plan_pts=plan_pts, ransac_thresh=ransac_thresh
            )
            vw.write(_prepare_frame(canvas))
            wrote += 1

        vw.release()
        if wrote == 0:
            raise RuntimeError("No frames were written to the video. Check your inputs and draw=True.")
