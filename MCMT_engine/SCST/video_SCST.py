# /workspace/MCMT_engine/SCST/video_SCST.py
from __future__ import annotations
import os, cv2, numpy as np
from typing import List, Dict, Any, Optional, Tuple, Iterable
from time import perf_counter as _t

from __Detection.detection_api import DetectionAPI
from __Tracking.tracking_api import TrackerAPI
from MCMT_engine.core.homoGraphy import PlanProjector
from MCMT_engine.core.renderer import PlanRenderer


class Args:
    track_thresh = 0.5
    match_thresh = 0.5
    track_buffer = 60
    mot20 = False
    cpu_workers = 10
    chunk_sec = 10.0
    batch_size = 20


class videoSCST:
    def __init__(self, plan_path: str, args: Optional[Args] = None,
                 detector: Optional[DetectionAPI] = None,
                 tracker: Optional[TrackerAPI] = None,
                 det_models: Optional[List[str]] = None,
                 det_device: str = "cuda",
                 det_threshold: float = 0.0,
                 det_class_names: Optional[List[str]] = None,
                 det_exclude: Optional[List[str]] = None,
                 det_device_map: Optional[Dict[str, str]] = None,
                 det_use_async: bool = True,
                 det_max_workers: Optional[int] = None,
                 ransac_thresh: float = 3.0):
        self.plan_img_path = plan_path
        self.ransac_thresh = float(ransac_thresh)
        self.args = args if args is not None else Args()

        print(f"[SCST] __init__: plan_path={plan_path}")
        # Detector (주입 우선, 없으면 생성)
        if detector is None:
            print(f"[SCST] __init__: create DetectionAPI(device={det_device}, models={det_models}, async={det_use_async})")
            self.detector = DetectionAPI(
                thres=det_threshold, device=det_device, models=det_models,
                exclude=det_exclude, device_map=det_device_map,
                class_names=det_class_names, use_async=det_use_async,
                max_workers=det_max_workers,
            )
        else:
            print("[SCST] __init__: use injected detector")
            self.detector = detector

        # Tracker (주입 우선, 없으면 생성)
        if tracker is None:
            print("[SCST] __init__: create TrackerAPI")
            self.tracker = TrackerAPI(args=self.args, detector=self.detector)
        else:
            print("[SCST] __init__: use injected tracker")
            self.tracker = tracker

        # 캐시/상태
        self.projector: Optional[PlanProjector] = None
        self._last_H: Optional[np.ndarray] = None
        self._last_cam_pts: Optional[Tuple[Tuple[float, float], ...]] = None
        self._last_plan_pts: Optional[Tuple[Tuple[float, float], ...]] = None
        self._last_plan_path: Optional[str] = None

    def _ensure_projector(self, plan_img_path: Optional[str]) -> None:
        """plan_img_path가 변경되었을 때만 PlanProjector 새로 생성"""
        ppath = plan_img_path or self.plan_img_path
        if (self.projector is None) or (self._last_plan_path != ppath):
            print(f"[SCST] _ensure_projector: assign plan image ({ppath})")
            self._last_plan_path = ppath
            # projector는 calibrate에서 정확한 pts와 함께 생성

    def calibrate(self, cam_pts, plan_pts, ransac_thresh: Optional[float] = None,
                  plan_img_path: Optional[str] = None) -> np.ndarray:
        print("[SCST] calibrate: start")
        t0 = _t()
        rth = float(ransac_thresh) if ransac_thresh is not None else self.ransac_thresh
        if len(cam_pts) < 4 or len(plan_pts) < 4 or len(cam_pts) != len(plan_pts):
            raise ValueError("Need >= 4 and same-length cam_pts & plan_pts.")

        self._ensure_projector(plan_img_path)

        # pts 동일하면 재적합 생략
        cam_key = tuple(map(tuple, cam_pts))
        plan_key = tuple(map(tuple, plan_pts))
        if (self.projector is not None and
            self._last_cam_pts == cam_key and
            self._last_plan_pts == plan_key and
            self._last_H is not None):
            print(f"[SCST] calibrate: reuse H (skip fit) time={(_t()-t0):.3f}s")
            return self._last_H

        # projector가 없으면 지금 생성
        if self.projector is None:
            print("[SCST] calibrate: create PlanProjector + initial H fit")
            self.projector = PlanProjector(
                plan_path=plan_img_path or self.plan_img_path,
                image_pts=cam_pts,
                plan_pts=plan_pts,
                ransac_thresh=rth,
            )
            H = self.projector.H
        else:
            print("[SCST] calibrate: refit homography with new points")
            H, _ = self.projector.fit_homography(
                image_pts=cam_pts, plan_pts=plan_pts, ransac_thresh=rth
            )

        self._last_H = H
        self._last_cam_pts = cam_key
        self._last_plan_pts = plan_key
        print(f"[SCST] calibrate: done |H|={float(np.linalg.norm(H)):.4f} time={(_t()-t0):.3f}s")
        return H

    def _iter_result_frames(self, results, stride: int = 1) -> Iterable[Tuple[int, List[Dict[str, Any]]]]:
        """results(dict or list)를 정렬된 프레임 순으로 순회"""
        if isinstance(results, dict):
            for i in sorted(results.keys()):
                if stride > 1 and (i % stride != 0):
                    continue
                yield i, results[i]
        else:
            for i, dets in enumerate(results):
                if stride > 1 and (i % stride != 0):
                    continue
                yield i, dets

    def track_and_save(self, video_path: str,
                       cam_pts, plan_pts,
                       plan_img_path: Optional[str] = None,
                       camera_save_path: Optional[str] = None,
                       plan_save_path: Optional[str] = None,
                       cam_trail_len: int = 30,
                       ransac_thresh: Optional[float] = None,
                       *,
                       # 추가: 도면 저장 최적화 옵션
                       plan_stride: int = 1,          # 2~3으로 올리면 도면 저장 부하/용량 감소
                       inplace_clear: bool = True     # True면 results[fi]를 즉시 None으로 해제
                       ):
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        # 1) 캘리브레이션
        H = self.calibrate(cam_pts=cam_pts, plan_pts=plan_pts,
                           ransac_thresh=ransac_thresh,
                           plan_img_path=plan_img_path)

        # 2) 디코딩 + 디텍션 + 트래킹
        print(f"[SCST] tracking: begin (decode via iter_frames_parallel, detect+associate inside TrackerCore)")
        t1 = _t()
        if camera_save_path:
            print(f"[SCST] tracking: with camera visualization → {camera_save_path}")
            results = self.tracker.track_video(
                video_path=video_path, save_path=camera_save_path, trail_len=cam_trail_len
            )
        else:
            print("[SCST] tracking: no camera visualization")
            results = self.tracker._track_video(video_path)
        track_time = _t() - t1
        print(f"[SCST] tracking: done in {track_time:.3f}s, frames={len(results)}")

        # 3) 도면 투영 인코딩(쓰기)
        if plan_save_path:
            fps = self._get_fps(video_path)
            print(f"[SCST] plan-encode: open writer {plan_save_path} (fps={fps})")
            t2 = _t()

            renderer = PlanRenderer(
                plan_img_or_path=plan_img_path or self.plan_img_path,
                draw_trails=True,
                trail_len=cam_trail_len,
                trail_ttl=30,
            )
            h, w = renderer.plan.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            os.makedirs(os.path.dirname(plan_save_path) or ".", exist_ok=True)
            vw = cv2.VideoWriter(plan_save_path, fourcc, float(fps), (w, h))
            if not vw.isOpened():
                raise RuntimeError(f"VideoWriter open failed: {plan_save_path}")

            written = 0
            prog_step = max(1, int(0.05 * len(results))) if len(results) > 0 else 50  # 5% 단위 또는 50프레임
            print(f"[SCST] plan-encode: start streaming (stride={plan_stride}, inplace_clear={inplace_clear})")

            try:
                for fi, dets in self._iter_result_frames(results, stride=plan_stride):
                    if dets is None:
                        continue
                    items = self.projector.projection(dets)
                    canvas = renderer.render_frame(items)
                    vw.write(canvas)
                    written += 1

                    if inplace_clear:
                        try:
                            if isinstance(results, dict):
                                results[fi] = None
                            else:
                                results[fi] = None
                        except Exception:
                            pass

                    if written % prog_step == 0:
                        print(f"[SCST] plan-encode: progress {written} frames written")

            finally:
                vw.release()

            print(f"[SCST] plan-encode: done in {(_t()-t2):.3f}s, written={written}, out={plan_save_path}")

        return results

    def _get_fps(self, video_path: str) -> float:
        print(f"[SCST] _get_fps: read fps from {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[SCST] _get_fps: fallback=30.0 (cap open failed)")
            return 30.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not fps or np.isnan(fps) or fps <= 1e-3:
            print("[SCST] _get_fps: fallback=30.0 (invalid fps)")
            return 30.0
        return float(fps)

    def close(self):
        print("[SCST] close")
        if hasattr(self.detector, "close"):
            try:
                self.detector.close()
            except Exception:
                pass
