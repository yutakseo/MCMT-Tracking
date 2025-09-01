from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Any
import time
import cv2
import numpy as np

from __Detection.detection_api import DetectionAPI
from __Tracking.tracking_api import TrackerAPI
from tools.homo_graphy import PlanProjector
from stream import StreamCCTV


# ----------------------------
# Args
# ----------------------------
@dataclass
class Args:
    track_thresh: float = 0.5
    match_thresh: float = 0.5
    track_buffer: int = 60
    mot20: bool = False
    cpu_workers: int = 10


# ----------------------------
# Camera pipeline
# ----------------------------
class Camera:
    def __init__(
        self,
        track_args: Args,
        cctv_url: str,
        cord_plan: str,
        plan_benchmark: List[Tuple[float, float]],
        cam_pts: List[Tuple[float, float]],
        transport: str = "tcp",
        max_width: int = 1920,
        results_dir: str = "./results",
    ):
        """
        plan_benchmark: 도면 상 기준점 좌표들 (plan 좌표)
        cam_pts       : 영상 상 기준점 좌표들 (image 좌표)
        """
        self.args = track_args
        self.url = cctv_url
        self.plan_path = cord_plan
        self.plan_benchmark = plan_benchmark
        self.calibration_mark: List[Tuple[float, float]] = cam_pts

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # IO
        self.cctv = StreamCCTV(url=cctv_url, transport=transport, max_width=max_width).start()
        self.cctv.wait_ready(10.0)

        # Models
        self.detector = DetectionAPI()
        self.tracker = TrackerAPI(args=track_args, detector=self.detector)

        # Projector (도면 캔버스)
        self.projector = PlanProjector(
            plan_img_or_path=cord_plan,
            trail_len=60,
            trail_ttl=30,
            line_thickness=4,
            point_radius=10,
        )

        # Homography
        self.H, self.mask = self.projector.fit_homography(
            image_pts=self.calibration_mark,
            plan_pts=self.plan_benchmark,
            ransac_thresh=5.0,
        )
        if self.H is None:
            raise RuntimeError("Homography estimation failed. Check correspondence points.")

        self._running = False

    # ----------------------------
    # Calibration update
    # ----------------------------
    def calibration(self, cam_pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        self.calibration_mark = cam_pts
        self.H, self.mask = self.projector.fit_homography(
            image_pts=self.calibration_mark,
            plan_pts=self.plan_benchmark,
            ransac_thresh=5.0,
        )
        if self.H is None:
            raise RuntimeError("Homography re-estimation failed.")
        return self.calibration_mark

    # ----------------------------
    # Capture raw frame
    # ----------------------------
    def capture(self) -> np.ndarray:
        frame = self.cctv.shoot()
        if frame is None:
            raise RuntimeError("Failed to read frame from stream.")
        return frame

    # ----------------------------
    # HUD
    # ----------------------------
    def _put_hud(self, img: Optional[np.ndarray], fps: float, n_tracks: int, label: str) -> None:
        if img is None:
            return
        txt = f"{label} | FPS: {fps:.1f} | Tracks: {n_tracks}"
        cv2.putText(img, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # ----------------------------
    # Snapshot (수동 저장)
    # ----------------------------
    def save_snapshot(self, img: np.ndarray, stem: str) -> Path:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = self.results_dir / f"{stem}_{ts}.png"
        cv2.imwrite(str(out_path), img)
        return out_path

    # ----------------------------
    # Keyboard handling
    # ----------------------------
    @staticmethod
    def _handle_key(paused: bool) -> tuple[bool, Optional[str]]:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return paused, "quit"
        if key == ord(' '):
            return not paused, None   # toggle pause
        if key == ord('s'):
            return paused, "snapshot"
        return paused, None

    # ----------------------------
    # Stream camera frames
    # ----------------------------
    def stream_camera(
        self,
        window_name: str = "Camera Stream",
        draw_tracks_on_frame: bool = False,
        target_fps: Optional[float] = None,
    ) -> None:
        self._running = True
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        paused = False
        t_last = time.time()
        fps = 0.0

        try:
            while self._running:
                if paused:
                    paused, action = self._handle_key(paused)
                    if action == "quit":
                        break
                    continue

                t0 = time.time()
                frame = self.capture()
                tracklets = self.tracker.track_image(frame=frame)

                if draw_tracks_on_frame and hasattr(self.tracker, "draw_on_frame"):
                    frame = self.tracker.draw_on_frame(frame, tracklets)

                # FPS
                now = time.time()
                dt = now - t_last
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
                t_last = now

                # Render
                n_tracks = len(tracklets) if hasattr(tracklets, "__len__") else 0
                view = frame.copy()
                self._put_hud(view, fps, n_tracks, label="Camera")
                cv2.imshow(window_name, view)

                paused, action = self._handle_key(paused)
                if action == "quit":
                    break
                if action == "snapshot":
                    self.save_snapshot(frame, stem="camera_snap")

                # (옵션) 표시 FPS 제한
                if target_fps:
                    spent = time.time() - t0
                    to_wait = max(0.0, (1.0 / target_fps) - spent)
                    if to_wait > 0:
                        time.sleep(to_wait)

        except KeyboardInterrupt:
            pass
        finally:
            self.close(window_name)

    # ----------------------------
    # Stream plan view
    # ----------------------------
    def stream_plan(
        self,
        window_name: str = "Plan Stream",
        draw_on_plan: bool = True,
        target_fps: Optional[float] = None,
    ) -> None:
        self._running = True
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

        paused = False
        t_last = time.time()
        fps = 0.0

        try:
            while self._running:
                if paused:
                    paused, action = self._handle_key(paused)
                    if action == "quit":
                        break
                    continue

                t0 = time.time()
                frame = self.capture()
                tracklets = self.tracker.track_image(frame=frame)

                projected, canvas = self.projector.projection(
                    dets_frame=tracklets,
                    mode="bottom-center",
                    draw=draw_on_plan,
                )

                now = time.time()
                dt = now - t_last
                if dt > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps > 0 else (1.0 / dt)
                t_last = now

                if canvas is not None:
                    n_tracks = len(tracklets) if hasattr(tracklets, "__len__") else 0
                    view = canvas.copy()
                    self._put_hud(view, fps, n_tracks, label="Plan")
                    cv2.imshow(window_name, view)

                    paused, action = self._handle_key(paused)
                    if action == "quit":
                        break
                    if action == "snapshot":
                        self.save_snapshot(canvas, stem="plan_snap")

                if target_fps:
                    spent = time.time() - t0
                    to_wait = max(0.0, (1.0 / target_fps) - spent)
                    if to_wait > 0:
                        time.sleep(to_wait)

        except KeyboardInterrupt:
            pass
        finally:
            self.close(window_name)

    # ----------------------------
    # Shutdown
    # ----------------------------
    def close(self, window_name: Optional[str] = None) -> None:
        self._running = False
        try:
            if hasattr(self.cctv, "stop"):
                self.cctv.stop()
        except Exception:
            pass
        if window_name is not None:
            try:
                cv2.destroyWindow(window_name)
            except Exception:
                pass
        cv2.destroyAllWindows()


# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":
    args = Args()

    cam_pts = [(1390, 521), (1618, 552), (1784, 578), (1112, 564)]
    plan_pts = [(100, 200), (300, 210), (320, 400), (90, 380)]

    cam = Camera(
        track_args=args,
        cctv_url="rtsp://user:pass@ip/stream",
        cord_plan="/path/to/plan.png",
        plan_benchmark=plan_pts,
        cam_pts=cam_pts,
        results_dir="./results",
    )

    # 원본 카메라 스트리밍
    # cam.stream_camera(draw_tracks_on_frame=True, target_fps=30.0)

    # 도면 스트리밍
    cam.stream_plan(draw_on_plan=True, target_fps=30.0)
