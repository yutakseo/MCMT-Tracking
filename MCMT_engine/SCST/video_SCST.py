# /workspace/MCMT_engine/SCST/video_SCST.py
from __future__ import annotations
import os, cv2, numpy as np
from typing import List, Dict, Any, Optional, Tuple

from __Detection.detection_api import DetectionAPI
from __Tracking.tracking_api import TrackerAPI
from MCMT_engine.core.homoGraphy import PlanProjector


class Args:
    track_thresh = 0.5
    match_thresh = 0.5
    track_buffer = 60
    mot20 = False
    cpu_workers = 10
    chunk_sec = 10.0
    batch_size = 20


class videoSCST:
    """
    단일 카메라 파이프라인:
      - Detection(내장 또는 외부 주입)
      - Tracking(+카메라뷰 저장) → TrackerAPI
      - Homography(도면 투영) → PlanProjector (시각화/저장은 내부 save_video 사용)

    cam_pts / plan_pts는 인스턴스 상태로 보관하지 않고,
    매 호출(calibrate/track_and_save) 시 인자로 받습니다.
    """
    def __init__(
        self,
        plan_path: str,
        args: Optional[Args] = None,

        # 외부 주입 가능 (주입 시 내부 생성 생략)
        detector: Optional[DetectionAPI] = None,
        tracker: Optional[TrackerAPI] = None,

        # 내부 생성 옵션 (외부 주입이 없을 때만 사용)
        det_models: Optional[List[str]] = None,
        det_device: str = "cuda",
        det_threshold: float = 0.0,
        det_class_names: Optional[List[str]] = None,
        det_exclude: Optional[List[str]] = None,
        det_device_map: Optional[Dict[str, str]] = None,
        det_use_async: bool = True,
        det_max_workers: Optional[int] = None,

        # 호모그래피 추정 파라미터
        ransac_thresh: float = 3.0,
    ):
        self.plan_img_path = plan_path
        self.ransac_thresh = float(ransac_thresh)
        self.args = args if args is not None else Args()

        # Detector
        if detector is not None:
            self.detector = detector
        else:
            self.detector = DetectionAPI(
                thres=det_threshold,
                device=det_device,
                models=det_models,
                exclude=det_exclude,
                device_map=det_device_map,
                class_names=det_class_names,
                use_async=det_use_async,
                max_workers=det_max_workers,
            )

        # Tracker
        if tracker is not None:
            self.tracker = tracker
        else:
            self.tracker = TrackerAPI(args=self.args, detector=self.detector)

        # ✅ PlanProjector는 이미지 경로만 받도록 생성
        self.projector = None
        self._last_H: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────
    # 캘리브레이션 (cam_pts/plan_pts 인자로 받아 H 추정)
    # ─────────────────────────────────────────────────────────
    def calibrate(
        self,
        cam_pts: List[Tuple[float, float]],
        plan_pts: List[Tuple[float, float]],
        ransac_thresh: Optional[float] = None,
        plan_img_path: Optional[str] = None,
    ) -> np.ndarray:
        if plan_img_path:
            # ✅ 시그니처에 맞게 생성자 교체
            self.projector = PlanProjector(plan_path=plan_img_path, plan_pts=plan_pts, image_pts=cam_pts)
            self.plan_img_path = plan_img_path

        rth = float(ransac_thresh) if ransac_thresh is not None else self.ransac_thresh

        if len(cam_pts) < 4 or len(plan_pts) < 4 or len(cam_pts) != len(plan_pts):
            raise ValueError("Need >= 4 and same-length cam_pts & plan_pts.")

        H, _ = self.projector.fit_homography(
            image_pts=cam_pts,
            plan_pts=plan_pts,
            ransac_thresh=rth,
        )
        self._last_H = H
        return H

    # ─────────────────────────────────────────────────────────
    # 추적/저장 (cam_pts/plan_pts 인자로 전달)
    # ─────────────────────────────────────────────────────────
    def track_and_save(
        self,
        video_path: str,
        cam_pts: List[Tuple[float, float]],
        plan_pts: List[Tuple[float, float]],

        plan_img_path: Optional[str] = None,
        camera_save_path: Optional[str] = None,   # 카메라뷰 저장 경로(선택)
        plan_save_path: Optional[str] = None,     # 도면 투영 저장 경로(선택)
        cam_trail_len: int = 30,
        ransac_thresh: Optional[float] = None,
    ) -> List[List[Dict[str, Any]]]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        # 호모그래피 계산/갱신
        H = self.calibrate(
            cam_pts=cam_pts,
            plan_pts=plan_pts,
            ransac_thresh=ransac_thresh,
            plan_img_path=plan_img_path,
        )
        print(f"[INFO] Camera calibration completed. H:\n{H}")

        # 카메라 뷰 저장 or 결과만
        if camera_save_path:
            results = self.tracker.track_video(
                video_path=video_path,
                save_path=camera_save_path,
                trail_len=cam_trail_len,
            )
        else:
            results = self.tracker._track_video(video_path)

        # 도면 투영 저장 (PlanProjector.save_video 시그니처에 맞춤)
        if plan_save_path:
            fps = self._get_fps(video_path)
            try:
                # 최신 분리 버전(plan_projector.py)의 시그니처:
                # save_video(frames, out_path, fps=30, image_pts=None, plan_pts=None, ransac_thresh=3.0)
                self.projector.save_video(
                    frames=results,
                    out_path=plan_save_path,
                    fps=float(fps),
                    image_pts=cam_pts,
                    plan_pts=plan_pts,
                    ransac_thresh=float(ransac_thresh) if ransac_thresh is not None else self.ransac_thresh,
                )
            except TypeError:
                # 구버전(모드 인자 등) 호환 처리
                self.projector.save_video(results, plan_save_path, fps=float(fps))
            print(f"[INFO] Plan projection saved: {plan_save_path}")

        return results

    # ─────────────────────────────────────────────────────────
    # 내부: FPS 추출
    # ─────────────────────────────────────────────────────────
    def _get_fps(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 30.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return 30.0 if not fps or np.isnan(fps) or fps <= 1e-3 else float(fps)

    # ─────────────────────────────────────────────────────────
    # 정리
    # ─────────────────────────────────────────────────────────
    def close(self):
        if hasattr(self.detector, "close"):
            try:
                self.detector.close()
            except Exception:
                pass
