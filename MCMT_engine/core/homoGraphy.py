# /workspace/MCMT_engine/core/homoGraphy.py
import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

# ─────────────────────────────────────────────────────────
# OpenCV 병렬/최적화 초기화
#   - 기본 스레드 수: 24 (OPENCV_NUM_THREADS가 있으면 그 값 우선)
#   - OpenCL 기본 OFF, 최적화 ON
# ─────────────────────────────────────────────────────────
def _init_cv_parallel() -> None:
    try:
        # 최적화 루틴 사용 (기본 ON)
        use_opt = os.getenv("OPENCV_USE_OPTIMIZED", "1") not in ("0", "false", "False")
        cv2.setUseOptimized(bool(use_opt))
    except Exception:
        pass

    try:
        # 스레드 수: env 없으면 기본 24
        th_env = os.getenv("OPENCV_NUM_THREADS", "").strip()
        th = int(th_env) if th_env else 24
        if th > 0:
            cv2.setNumThreads(th)
    except Exception:
        pass

    try:
        # OpenCL 기본 OFF (충돌 방지)
        use_ocl = os.getenv("OPENCV_USE_OPENCL", "0") in ("1", "true", "True")
        if hasattr(cv2, "ocl"):
            cv2.ocl.setUseOpenCL(bool(use_ocl))
    except Exception:
        pass


# 투영 결과 스키마:
# {
#   "id": Optional[int],
#   "cls": Optional[str],        # label 또는 class
#   "bbox": List[float],         # xyxy (x1,y1,x2,y2)
#   "pt": Tuple[float, float],   # 도면 좌표 (바닥 중심 투영)
# }

class PlanProjector:
    """
    도면(평면) 위로 검출/추적 결과의 '바닥 중심(bottom-center)'을
    호모그래피로 투영해 도면 좌표를 계산하는 유틸(시각화 없음).

    초기화 시:
      - plan_path 로 도면 이미지를 로드
      - image_pts ↔ plan_pts 로 호모그래피 H를 적합(fit)
    """

    def __init__(
        self,
        plan_path: Union[str, np.ndarray],
        image_pts: List[Tuple[float, float]],
        plan_pts: List[Tuple[float, float]],
        ransac_thresh: float = 3.0,
    ) -> None:
        # OpenCV 병렬/최적화 초기화 (전역)
        _init_cv_parallel()

        # 1) 도면 이미지 로드
        if isinstance(plan_path, str):
            plan = cv2.imread(plan_path)
            if plan is None:
                raise FileNotFoundError(f"Cannot read plan image: {plan_path}")
            self.plan = plan
            self.plan_path = plan_path
        else:
            self.plan = plan_path.copy()
            self.plan_path = None

        # 2) 호모그래피 적합
        self.H, _ = self.fit_homography(image_pts, plan_pts, ransac_thresh)

    # -------------------- Homography --------------------
    def fit_homography(
        self,
        image_pts: Optional[List[Tuple[float, float]]],
        plan_pts: List[Tuple[float, float]],
        ransac_thresh: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(image_pts ↔ plan_pts)로 H를 계산하고 내부에 설정."""
        ip = np.asarray(image_pts, np.float32)
        pp = np.asarray(plan_pts,  np.float32)
        if ip.shape[0] < 4 or pp.shape[0] < 4 or ip.shape[0] != pp.shape[0]:
            raise ValueError("Need >= 4 correspondences and same length for image_pts and plan_pts.")
        H, mask = cv2.findHomography(ip, pp, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if H is None:
            raise RuntimeError("findHomography failed.")
        self.H = H
        return H, mask
    
    def projection_from_xyxy(self, boxes_xyxy: np.ndarray) -> np.ndarray:
        """
        Fast path: 이미 xyxy(float32, (N,4)) 박스 배열이 있을 때,
        bottom-center만 추출해서 H로 투영해 (N,2) 도면 좌표를 반환.
        """
        if boxes_xyxy is None:
            return np.empty((0, 2), dtype=np.float32)
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if boxes.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = boxes[:, 3]  # bottom y
        pts = np.stack([cx, cy], axis=1)  # (N,2)
        return self._project_points(self.H, pts)  # (N,2)


    # -------------------- Internals --------------------
    @staticmethod
    def _xyxy_from_any(bbox: Union[List[float], np.ndarray]) -> np.ndarray:
        """bbox를 xyxy로 표준화 (tlwh도 허용)."""
        b = np.asarray(bbox, dtype=np.float32).reshape(-1)
        if b.shape[0] < 4:
            raise ValueError("bbox length must be >= 4")

        x1, y1, a, b2 = b[0], b[1], b[2], b[3]
        if a > x1 and b2 > y1:  # already xyxy
            return np.array([x1, y1, a, b2], dtype=np.float32)
        if a > 0 and b2 > 0:    # tlwh
            return np.array([x1, y1, x1 + a, y1 + b2], dtype=np.float32)
        x2, y2 = a, b2
        return np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)

    @staticmethod
    def _project_points(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
        """(N,2) pts_xy를 (3x3) H로 투영하여 (N,2) 반환"""
        pts = np.asarray(pts_xy, dtype=np.float32).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(pts, H)
        return proj.reshape(-1, 2)

    # -------------------- Public API --------------------
    def projection(
        self,
        dets_frame: List[Dict],
    ) -> List[Dict]:
        """
        단일 프레임 투영 (시각화 없음).
        입력: dets_frame = [{'id':..., 'bbox':..., 'label' or 'class': ...}, ...]
        출력: [{'id':.., 'cls':.., 'bbox':[x1,y1,x2,y2], 'pt':(X,Y)}, ...]
        """
        if self.H is None:
            raise RuntimeError("Homography not set. Failed initialization.")

        # 벡터화: bottom-center 좌표를 한 번에 투영
        boxes, meta = [], []
        for d in dets_frame:
            bbox = d.get("bbox", d.get("box", d.get("tlwh", None)))
            if bbox is None:
                continue
            x1, y1, x2, y2 = self._xyxy_from_any(bbox)
            boxes.append((x1, y1, x2, y2))
            meta.append(d)

        if not boxes:
            return []

        boxes = np.asarray(boxes, dtype=np.float32)
        cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
        cy = boxes[:, 3]  # bottom y
        pts = np.stack([cx, cy], axis=1)  # [N,2]

        proj = self._project_points(self.H, pts)  # [N,2]

        out: List[Dict] = []
        for i, d in enumerate(meta):
            x1, y1, x2, y2 = boxes[i]
            p = proj[i]
            item = {
                "id": int(d["id"]) if "id" in d and d["id"] is not None else None,
                "cls": d.get("label", d.get("class", None)),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "pt": (float(p[0]), float(p[1])),
            }
            out.append(item)
        return out

    def project_video(
        self,
        frames: List[List[Dict]],
    ) -> List[List[Dict]]:
        """여러 프레임 일괄 투영 (시각화 없음)."""
        return [self.projection(f) for f in frames]
