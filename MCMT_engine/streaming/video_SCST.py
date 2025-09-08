# /workspace/MCMT_engine/scst_camera.py
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
        # --- 필수: 도면 정보 ---
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
        # 도면 정보 저장
        self.plan_img_path = plan_img_path
        self.plan_pts = plan_pts
        self.ransac_thresh = ransac_thresh
        
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

        # Projector & Homography (도면 정보만 초기화, 켈리브레이션은 별도 함수에서)
        self.projector = PlanProjector(
            plan_img_or_path=plan_img_path,
            trail_len=plan_trail_len,
            trail_ttl=plan_trail_ttl,
            line_thickness=plan_line_thickness,
            point_radius=plan_point_radius,
        )

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
    # 핵심 API: 켈리브레이션 + 비디오 → 추적/저장 + 도면(미니맵) 저장 (한 큐에 처리)
    # ─────────────────────────────────────────────────────────
    def track_and_save(
        self,
        video_path: str,
        cctv_pts: List[Tuple[float, float]],  # CCTV 영상 내 기준점 좌표 리스트
        camera_save_path: Optional[str] = None,  # 카메라 영상 결과 저장 경로 (None이면 저장 안 함)
        plan_save_path: Optional[str] = None,    # 도면(미니맵) 결과 저장 경로 (None이면 저장 안 함)
        plan_mode: str = "bottom-center",
        cam_trail_len: int = 30,                 # 카메라 영상에 그릴 궤적 길이
        ransac_thresh: Optional[float] = None,   # RANSAC 임계값 (None이면 생성자에서 설정한 값 사용)
    ) -> List[List[Dict[str, Any]]]:
        """
        켈리브레이션 + 비디오 처리 및 추적 수행 (한 큐에 처리)
        
        Args:
            video_path: 처리할 비디오 파일 경로
            cctv_pts: CCTV 영상 내 기준점 좌표 리스트
            camera_save_path: 카메라 영상 결과 저장 경로
            plan_save_path: 도면(미니맵) 결과 저장 경로
            plan_mode: 도면 렌더링 모드
            cam_trail_len: 카메라 영상에 그릴 궤적 길이
            ransac_thresh: RANSAC 임계값
            
        Returns:
            List[List[Dict[str, Any]]]: 추적 결과
        """
        # 1. 켈리브레이션 수행
        if ransac_thresh is None:
            ransac_thresh = self.ransac_thresh
            
        try:
            H, _ = self.projector.fit_homography(cctv_pts, self.plan_pts, ransac_thresh=ransac_thresh)
            print(f"[INFO] Camera calibration completed successfully")
        except Exception as e:
            print(f"[ERROR] Camera calibration failed: {e}")
            raise RuntimeError(f"Camera calibration failed: {e}")
        
        # 2. 비디오 파일 확인
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path does not exist: {video_path}")

        # 3. 비디오별 상태 초기화
        self.core.reset_tracker()
        self.core.img_size = None
        self.visualizer.reset()
        results: List[List[Dict[str, Any]]] = []

        # 4. 카메라 뷰 저장 준비 (선택)
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

        # 5. 병렬 디코딩 → 순서 보정 → 배치 처리
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
                    # 6. 객체 탐지 + 추적
                    batch_res = self.core.track_video_batch(batch_frames)
                    
                    # 7. 카메라 영상 저장 (선택)
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

        # 8. 도면(미니맵) 저장 (선택) - 호모그래피 적용
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
    # 선택: H 갱신
    # ─────────────────────────────────────────────────────────
    def refit_homography(
        self,
        cctv_pts: List[Tuple[float, float]],
        plan_pts: Optional[List[Tuple[float, float]]] = None,
        ransac_thresh: Optional[float] = None,
    ):
        """
        호모그래피 재계산
        
        Args:
            cctv_pts: CCTV 영상 내 기준점 좌표 리스트
            plan_pts: 도면 내 기준점 좌표 리스트 (None이면 기존 값 사용)
            ransac_thresh: RANSAC 임계값 (None이면 기존 값 사용)
        """
        if plan_pts is not None:
            self.plan_pts = plan_pts
        if ransac_thresh is not None:
            self.ransac_thresh = ransac_thresh
            
        self.cctv_pts = cctv_pts
        self.H, _ = self.projector.fit_homography(self.cctv_pts, self.plan_pts, ransac_thresh=self.ransac_thresh)

    # ─────────────────────────────────────────────────────────
    # 정리(선택)
    # ─────────────────────────────────────────────────────────
    def close(self):
        if hasattr(self.detector, "close"):
            try: self.detector.close()
            except Exception: pass
        if hasattr(self.visualizer, "reset"):
            self.visualizer.reset()