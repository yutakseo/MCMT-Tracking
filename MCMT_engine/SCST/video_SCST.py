# /workspace/MCMT_engine/SCST/video_SCST.py
from __future__ import annotations
import os, cv2, numpy as np, threading
from typing import List, Dict, Any, Optional, Tuple, Iterable
from time import perf_counter as _t

from __Detection.detection_api import DetectionAPI
from __Tracking.tracking_api import TrackerAPI
from MCMT_engine.core.homoGraphy import PlanProjector
from MCMT_engine.core.renderer import PlanRenderer

# ★ 추가: 온라인 투영 파이프라인
from MCMT_engine.core.online_projection import OnlineProjectionPipeline, ProjItem


class Args:
    # --- ByteTrack / 매칭 ---
    track_thresh = 0.5
    match_thresh = 0.5
    track_buffer = 60
    mot20 = False

    # --- 배치 / 추론 ---
    batch_size = 20

    # --- 디코더(TrackerAPI에서 사용) ---
    # decode_threads가 None이면 cpu_workers를 사용(하위호환)
    cpu_workers = 10
    decode_threads = None         # 권장: 0(FFmpeg 자동) 또는 CPU 상황에 맞게 정수
    prefetch_frames = 512         # 128~4096 사이에서 메모리 상황에 맞게
    hwaccel = None                # 'cuda' / None
    decode_target_size = None     # (width, height) 지정 시 디코더에서 다운스케일

    # --- 구버전 호환(미사용) ---
    chunk_sec = 10.0


class videoSCST:
    """
    - calibrate() : 카메라-도면 간 호모그래피 추정/캐시
    - track_and_save() : TrackerAPI를 사용해 추적(카메라 비디오 저장 optional) + 도면 위 렌더링 optional
      ※ plan_save_path가 지정되면 '온라인 투영/인코딩'으로 진행(배치마다 즉시 반영)
    """
    def __init__(
        self,
        plan_path: str,
        args: Optional[Args] = None,
        detector: Optional[DetectionAPI] = None,
        tracker: Optional[TrackerAPI] = None,
        *,
        det_models: Optional[List[str]] = None,
        det_device: str = "cuda",
        det_threshold: float = 0.0,
        det_class_names: Optional[List[str]] = None,
        det_exclude: Optional[List[str]] = None,
        det_device_map: Optional[Dict[str, str]] = None,
        det_use_async: bool = True,
        det_max_workers: Optional[int] = None,
        ransac_thresh: float = 3.0,
    ):
        self.plan_img_path = plan_path
        self.ransac_thresh = float(ransac_thresh)
        self.args = args if args is not None else Args()

        print(f"[SCST] __init__: plan_path={plan_path}")

        # --- Detector (주입 우선) ---
        if detector is None:
            print(f"[SCST] __init__: create DetectionAPI(device={det_device}, models={det_models}, async={det_use_async})")
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
        else:
            print("[SCST] __init__: use injected detector")
            self.detector = detector

        # --- Tracker (주입 우선) ---
        if tracker is None:
            print("[SCST] __init__: create TrackerAPI")
            self.tracker = TrackerAPI(args=self.args, detector=self.detector)
        else:
            print("[SCST] __init__: use injected tracker")
            self.tracker = tracker

        # --- 상태/캐시 ---
        self.projector: Optional[PlanProjector] = None
        self._last_H: Optional[np.ndarray] = None
        self._last_cam_pts: Optional[Tuple[Tuple[float, float], ...]] = None
        self._last_plan_pts: Optional[Tuple[Tuple[float, float], ...]] = None
        self._last_plan_path: Optional[str] = None

    # ----------------------------
    # 내부 유틸
    # ----------------------------
    def _ensure_projector(self, plan_img_path: Optional[str]) -> None:
        """plan_img_path가 변경되었을 때만 PlanProjector 새로 생성(실제 생성은 calibrate에서)"""
        ppath = plan_img_path or self.plan_img_path
        if (self.projector is None) or (self._last_plan_path != ppath):
            print(f"[SCST] _ensure_projector: assign plan image ({ppath})")
            self._last_plan_path = ppath
            # projector 생성은 calibrate()에서 pts와 함께 수행

    def _iter_result_frames(self, results, stride: int = 1) -> Iterable[Tuple[int, List[Dict[str, Any]]]]:
        """(오프라인 경로용) results(dict or list)를 정렬된 프레임 순으로 순회"""
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

    # ----------------------------
    # 공개 API
    # ----------------------------
    def calibrate(
        self,
        cam_pts,
        plan_pts,
        ransac_thresh: Optional[float] = None,
        plan_img_path: Optional[str] = None,
    ) -> np.ndarray:
        print("[SCST] calibrate: start")
        t0 = _t()
        rth = float(ransac_thresh) if ransac_thresh is not None else self.ransac_thresh
        if len(cam_pts) < 4 or len(plan_pts) < 4 or len(cam_pts) != len(plan_pts):
            raise ValueError("Need >= 4 and same-length cam_pts & plan_pts.")

        self._ensure_projector(plan_img_path)

        # pts 동일하면 재적합 생략
        cam_key = tuple(map(tuple, cam_pts))
        plan_key = tuple(map(tuple, plan_pts))
        if (
            self.projector is not None
            and self._last_cam_pts == cam_key
            and self._last_plan_pts == plan_key
            and self._last_H is not None
        ):
            print(f"[SCST] calibrate: reuse H (skip fit) time={(_t()-t0):.3f}s")
            return self._last_H

        # projector 생성/재적합
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

    def track_and_save(
        self,
        video_path: str,
        *,
        cam_pts,
        plan_pts,
        plan_img_path: Optional[str] = None,
        camera_save_path: Optional[str] = None,
        plan_save_path: Optional[str] = None,
        cam_trail_len: int = 30,
        ransac_thresh: Optional[float] = None,
        # 도면 저장 최적화
        plan_stride: int = 1,          # 2~3으로 올리면 도면 저장 부하/용량 감소
        inplace_clear: bool = True,    # (오프라인 경로 전용) True면 results[fi]를 즉시 None으로 해제
    ):
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        # 1) 캘리브레이션 (PlanProjector 준비)
        self.calibrate(
            cam_pts=cam_pts,
            plan_pts=plan_pts,
            ransac_thresh=ransac_thresh,
            plan_img_path=plan_img_path,
        )

        # ─────────────────────────────────────────────────────────
        # 2) (선택) 온라인 도면 투영/인코딩 준비
        # ─────────────────────────────────────────────────────────
        online = None
        vw = None
        renderer = None
        written = 0
        consumer_th: Optional[threading.Thread] = None

        if plan_save_path:
            fps = self._get_fps(video_path)
            print(f"[SCST] plan-encode(online): open writer {plan_save_path} (fps={fps})")

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

            # 2-1) 온라인 투영 파이프라인
            online = OnlineProjectionPipeline(self.projector, max_queue=1024)

            # 2-2) 소비자(순서보장 emit) 정의
            def emit_to_plan_encoder(item: ProjItem):
                nonlocal written
                # stride 적용
                if (item.frame_id % int(plan_stride)) != 0:
                    return

                pts = item.pts_plan
                n = len(pts)
                ids = (item.ids.tolist() if isinstance(item.ids, np.ndarray) else (item.ids or [None]*n))
                clss = item.clss or [None]*n

                det_like = []
                for i in range(n):
                    det_like.append({
                        "id": int(ids[i]) if ids[i] is not None else None,
                        "cls": clss[i],
                        "bbox": item.boxes_xyxy[i].astype(float).tolist(),
                        "pt": (float(pts[i,0]), float(pts[i,1])),
                    })
                canvas = renderer.render_frame(det_like)
                vw.write(canvas)
                written += 1
                if written % 100 == 0:
                    print(f"[SCST] plan-encode(online): progress {written} frames written")

            # 2-3) 소비 루프 시작 (별 스레드)
            consumer_th = threading.Thread(
                target=online.consume_and_emit,
                args=(emit_to_plan_encoder, 0),  # 시작 frame_id가 0이 아닐 경우 조정
                name="PlanEmit",
                daemon=True,
            )
            online.start()
            consumer_th.start()

            # 2-4) Tracker에 전달할 콜백
            def _on_det(frame_id, boxes_xyxy, ids, classes, meta):
                online.on_detection(frame_id, boxes_xyxy, ids, classes, meta)
        else:
            _on_det = None

        # ─────────────────────────────────────────────────────────
        # 3) 디코딩+디텍션+트래킹 (카메라 영상 저장 여부 선택)
        # ─────────────────────────────────────────────────────────
        print("[SCST] tracking: begin (streaming decoder → detect & associate in TrackerCore)")
        t1 = _t()
        if camera_save_path:
            print(f"[SCST] tracking: with camera visualization → {camera_save_path}")
            results = self.tracker.track_video(
                video_path=video_path, save_path=camera_save_path, trail_len=cam_trail_len,
                on_detection=_on_det,
            )
        else:
            print("[SCST] tracking: no camera visualization")
            # 가능하면 track_video를 통일해서 on_detection 사용, 여기선 저장 없으니 save_path=None
            results = self.tracker.track_video(
                video_path=video_path, save_path=None, trail_len=cam_trail_len,
                on_detection=_on_det,
            )
        track_time = _t() - t1
        print(f"[SCST] tracking: done in {track_time:.3f}s, frames={len(results)}")

        # ─────────────────────────────────────────────────────────
        # 4) 마무리 (온라인 경로 정리)
        # ─────────────────────────────────────────────────────────
        if online is not None:
            online.stop_and_join()
            if consumer_th is not None and consumer_th.is_alive():
                consumer_th.join(timeout=1.0)
            if vw is not None:
                vw.release()
                print(f"[SCST] plan-encode(online): done, written={written}, out={plan_save_path}")

        # (참고) 오프라인 경로가 필요하면 아래 블록을 활성화하면 됨:
        # if plan_save_path and _on_det is None:
        #     self._encode_plan_offline(results, video_path, plan_img_path, cam_trail_len, plan_save_path, plan_stride, inplace_clear)

        return results

    # (옵션) 오프라인 인코딩을 별도 메서드로 빼두면 유지보수 편함
    def _encode_plan_offline(
        self, results, video_path, plan_img_path, cam_trail_len, plan_save_path, plan_stride, inplace_clear
    ):
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
        prog_step = max(1, int(0.05 * len(results))) if len(results) > 0 else 50
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

    def close(self):
        print("[SCST] close")
        if hasattr(self.detector, "close"):
            try:
                self.detector.close()
            except Exception:
                pass
