# /workspace/MCMT_engine/streaming/video_SCST.py
from __future__ import annotations
import os, cv2, numpy as np
from typing import List, Dict, Any, Optional, Iterable, Tuple

from __Detection.detection_api import DetectionAPI
from __Tracking.core.tracker_core import TrackerCore
from __Tracking.utils.visualizer import TrackerVisualizer
from __Tracking.utils.video_decoder import iter_frames_parallel
from tools.homo_graphy import PlanProjector


class Args:
    # ByteTrack & I/O 기본값 (필요 시 외부에서 덮어쓰기)
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
      - Detection(내장/선택)
      - Tracking(ByteTrack)
      - Homography(도면 미니맵)
      - 카메라 영상/도면 영상 저장
    """
    def __init__(
        self,
        # --- 기본 도면 정보(디폴트) ---
        plan_img_path: str,
        plan_pts: List[Tuple[float, float]],

        # --- 추적 파라미터 ---
        args: Optional[Args] = None,

        # --- Detector 자동 내장(원하면 외부 주입 대신 자동 생성) ---
        det_models: Optional[List[str]] = None,          # ["ultra_people", "vehicle"] 등
        det_device: str = "cuda",
        det_threshold: float = 0.0,
        det_class_names: Optional[List[str]] = None,
        det_exclude: Optional[List[str]] = None,
        det_device_map: Optional[Dict[str, str]] = None, # {"ultra_people":"cuda:0", ...}
        det_use_async: bool = True,
        det_max_workers: Optional[int] = None,

        # --- 도면 렌더 옵션 ---
        plan_trail_len: int = 60,
        plan_trail_ttl: int = 30,
        plan_line_thickness: int = 4,
        plan_point_radius: int = 10,
        ransac_thresh: float = 3.0,
    ):
        # 디폴트 도면 정보
        self.plan_img_path = plan_img_path
        self.plan_pts = plan_pts
        self.ransac_thresh = ransac_thresh

        # projector 재생성 시 필요한 옵션들 캐시
        self._proj_opts = dict(
            trail_len=plan_trail_len,
            trail_ttl=plan_trail_ttl,
            line_thickness=plan_line_thickness,
            point_radius=plan_point_radius,
        )
        self._plan_img_cached_path = plan_img_path

        # args
        self.args = args if args is not None else Args()
        self.cpu_workers = int(getattr(self.args, "cpu_workers", 8))
        self.chunk_sec   = float(getattr(self.args, "chunk_sec", 20.0))
        self.batch_size  = int(getattr(self.args, "batch_size", 8))

        # Detector (내장)
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

        # Tracker/Visualizer
        self.core = TrackerCore(self.args, self.detector)
        self.visualizer = TrackerVisualizer()

        # Projector (도면 이미지 로드 1회)
        self.projector = PlanProjector(plan_img_or_path=plan_img_path, **self._proj_opts)

        # 최근 캘리브레이션 상태(로그/디버깅용)
        self._last_cctv_pts: Optional[List[Tuple[float, float]]] = None
        self._last_plan_pts: Optional[List[Tuple[float, float]]] = None
        self._last_H: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────
    # 내부: 병렬 디코딩 out-of-order → in-order 보정
    # ─────────────────────────────────────────────────────────
    def _ordered_stream(self, frame_stream: Iterable[Tuple[int, np.ndarray]]):
        buffer = {}
        expected = None
        for idx, frame in frame_stream:
            if frame is None or frame.size == 0:
                continue
            buffer[idx] = frame
            if expected is None:
                expected = idx
            while expected in buffer:
                f = buffer.pop(expected)
                yield expected, f
                expected += 1
        if buffer:
            for k in sorted(buffer.keys()):
                yield k, buffer[k]
        buffer.clear()

    # ─────────────────────────────────────────────────────────
    # 새 도면 이미지/포인트 설정 (인스턴스 재사용 시)
    # ─────────────────────────────────────────────────────────
    def set_plan(
        self,
        plan_img_path: Optional[str] = None,
        plan_pts: Optional[List[Tuple[float, float]]] = None,
    ):
        """도면 이미지/포인트 갱신. 이미지 바뀌면 Projector만 새로 만듦(Detector/Tracker 재사용)."""
        if plan_pts is not None:
            self.plan_pts = plan_pts

        if plan_img_path and (plan_img_path != self._plan_img_cached_path):
            # Projector만 교체 (가벼움)
            self.projector = PlanProjector(plan_img_or_path=plan_img_path, **self._proj_opts)
            self.plan_img_path = plan_img_path
            self._plan_img_cached_path = plan_img_path

    # ─────────────────────────────────────────────────────────
    # 캘리브레이션 전용 API (H만 다시 맞추고 싶을 때)
    # ─────────────────────────────────────────────────────────
    def calibrate(
        self,
        cctv_pts: List[Tuple[float, float]],
        plan_pts: Optional[List[Tuple[float, float]]] = None,
        ransac_thresh: Optional[float] = None,
        plan_img_path: Optional[str] = None,
    ) -> np.ndarray:
        """호모그래피 H만 재추정(인스턴스/검출기 재생성 없이)."""
        if plan_img_path or plan_pts is not None:
            self.set_plan(plan_img_path=plan_img_path, plan_pts=plan_pts)

        if ransac_thresh is None:
            ransac_thresh = self.ransac_thresh

        H, _ = self.projector.fit_homography(cctv_pts, self.plan_pts, ransac_thresh=ransac_thresh)

        # 상태 기록
        self._last_cctv_pts = list(cctv_pts)
        self._last_plan_pts = list(self.plan_pts)
        self._last_H = H
        return H

    # ─────────────────────────────────────────────────────────
    # 핵심 API: (옵션) 켈리브레이션 override + 추적/저장
    # ─────────────────────────────────────────────────────────
    def track_and_save(
        self,
        video_path: str,
        cctv_pts: List[Tuple[float, float]],

        # ⬇️ 호출마다 덮어쓰기(override) 가능
        plan_pts: Optional[List[Tuple[float, float]]] = None,
        plan_img_path: Optional[str] = None,

        camera_save_path: Optional[str] = None,  # 카메라 영상 결과 저장
        plan_save_path: Optional[str] = None,    # 도면(미니맵) 결과 저장
        plan_mode: str = "bottom-center",
        cam_trail_len: int = 30,
        ransac_thresh: Optional[float] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        캘리브레이션(호출 시 지정 가능) + 비디오 처리 및 추적 수행
        - plan_pts / plan_img_path 를 넘기면 인스턴스 기본값을 덮어써서 사용
        """
        # 0) 입력 검증
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path does not exist: {video_path}")

        # 1) (선택) 도면/포인트 override 반영
        self.set_plan(plan_img_path=plan_img_path, plan_pts=plan_pts)

        # 2) H 추정 (매 호출마다 다른 매핑으로)
        if ransac_thresh is None:
            ransac_thresh = self.ransac_thresh
        try:
            H, _ = self.projector.fit_homography(cctv_pts, self.plan_pts, ransac_thresh=ransac_thresh)
            self._last_cctv_pts = list(cctv_pts)
            self._last_plan_pts = list(self.plan_pts)
            self._last_H = H
            print(f"[INFO] Camera calibration completed (points={len(cctv_pts)} -> {len(self.plan_pts)})")
        except Exception as e:
            print(f"[ERROR] Camera calibration failed: {e}")
            raise RuntimeError(f"Camera calibration failed: {e}")

        # 3) 추적기 초기화
        self.core.reset_tracker()
        self.core.img_size = None
        self.visualizer.reset()
        results: List[List[Dict[str, Any]]] = []

        # 4) 카메라 뷰 저장 준비 (선택)
        writer = None
        fps = 30.0
        if camera_save_path:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {video_path}")
            fps0 = cap.get(cv2.CAP_PROP_FPS)
            fps = fps0 if fps0 and not np.isnan(fps0) and fps0 > 1e-3 else 30.0
            width, height = int(cap.get(3)), int(cap.get(4))
            if width == 0 or height == 0:
                ret, f0 = cap.read()
                if not ret or f0 is None:
                    cap.release()
                    raise RuntimeError("Unable to determine video frame size")
                height, width = f0.shape[:2]
            cap.release()
            writer, camera_save_path = self._create_writer(camera_save_path, fps, width, height)

        # 5) 병렬 디코딩 → 순서 보정 → 배치 처리
        raw_stream = iter_frames_parallel(video_path, cpu_workers=self.cpu_workers, chunk_sec=self.chunk_sec)
        stream = self._ordered_stream(raw_stream)

        frame_count = 0
        batch_frames: List[np.ndarray] = []
        batch_raw: Optional[List[np.ndarray]] = [] if writer is not None else None

        try:
            for _, frame in stream:
                batch_frames.append(frame)
                if batch_raw is not None:
                    batch_raw.append(frame)

                if len(batch_frames) >= self.batch_size:
                    batch_res = self.core.track_video_batch(batch_frames)
                    if writer is not None:
                        for f, r in zip(batch_raw, batch_res):
                            vis = self.visualizer.draw_frame(f, r, trail_len=int(min(cam_trail_len, 1000)))
                            writer.write(vis)
                            frame_count += 1
                        batch_raw.clear()
                    results.extend(batch_res)
                    batch_frames.clear()

            # 남은 배치 처리
            if batch_frames:
                batch_res = self.core.track_video_batch(batch_frames)
                if writer is not None:
                    for f, r in zip(batch_raw, batch_res):
                        vis = self.visualizer.draw_frame(f, r, trail_len=int(min(cam_trail_len, 1000)))
                        writer.write(vis)
                        frame_count += 1
                    batch_raw.clear()
                results.extend(batch_res)
                batch_frames.clear()
        finally:
            if writer is not None:
                writer.release()

        if writer is not None:
            print(f"[INFO] Camera view saved: {camera_save_path}, frames: {frame_count}")

        # 6) 도면(미니맵) 저장 (선택) - 현재 projector 상태(H 포함)를 사용
        if plan_save_path:
            self.projector.save_video(results, plan_save_path, fps=float(fps), mode=plan_mode)
            print(f"[INFO] Plan projection saved: {plan_save_path}")

        return results

    # ─────────────────────────────────────────────────────────
    # 내부: VideoWriter 생성
    # ─────────────────────────────────────────────────────────
    def _create_writer(self, save_path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            alt_path = os.path.splitext(save_path)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(alt_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Failed to open VideoWriter with both mp4v and MJPG")
            save_path = alt_path
        return writer, save_path

    # ─────────────────────────────────────────────────────────
    # 호모그래피 재계산 (호출 시 플랜/스레숄드 변경 가능)
    # ─────────────────────────────────────────────────────────
    def refit_homography(
        self,
        cctv_pts: List[Tuple[float, float]],
        plan_pts: Optional[List[Tuple[float, float]]] = None,
        ransac_thresh: Optional[float] = None,
        plan_img_path: Optional[str] = None,
    ):
        """인스턴스 유지한 채로 H만 갱신하고 싶을 때 사용."""
        self.calibrate(
            cctv_pts=cctv_pts,
            plan_pts=plan_pts,
            ransac_thresh=ransac_thresh,
            plan_img_path=plan_img_path,
        )

    # ─────────────────────────────────────────────────────────
    # 정리(선택)
    # ─────────────────────────────────────────────────────────
    def close(self):
        if hasattr(self.detector, "close"):
            try: self.detector.close()
            except Exception: pass
        if hasattr(self.visualizer, "reset"):
            self.visualizer.reset()
