# /workspace/tools/visualizer.py
import cv2
import numpy as np
from typing import List, Tuple, Optional, Any

# 카메라별 기본 색상 (BGR)
DEFAULT_CAM_COLORS = [
    (80, 180, 255),   # cam1
    (120, 220, 60),   # cam2
    (255, 180, 90),   # cam3
    (200, 120, 255),  # cam4...
    (100, 150, 250),
    (180, 180, 180),
]
DEFAULT_FUSED_COLOR = (30, 30, 255)  # 빨강


def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:  # GRAY -> BGR
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:  # BGRA -> BGR
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def get_plan_image_from_cams(cams: List[Any]) -> Optional[np.ndarray]:
    """
    카메라 projector 안에서 플랜 이미지를 찾아 반환. 없으면 None.
    """
    for cam in cams:
        proj = getattr(cam, "projector", None)
        for attr in ("plan_img", "_plan_img", "img", "plan"):
            pl = getattr(proj, attr, None) if proj is not None else None
            if isinstance(pl, np.ndarray) and pl.ndim >= 2 and pl.size > 0:
                return _to_bgr(pl.copy())
    return None


def load_or_create_plan(base_img: Optional[np.ndarray], fallback_path: Optional[str], fallback_size=(2560, 1440)) -> np.ndarray:
    if isinstance(base_img, np.ndarray):
        return _to_bgr(base_img)
    if fallback_path:
        img = cv2.imread(fallback_path)
        if isinstance(img, np.ndarray):
            return _to_bgr(img)
    w, h = fallback_size
    return np.full((h, w, 3), 255, np.uint8)  # white


def draw_points(
    img: np.ndarray,
    pts: List[Tuple[float, float]],
    color: Tuple[int, int, int],
    radius: int = 5,
    thickness: int = -1,
    with_index: bool = False,
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    for i, (x, y) in enumerate(pts):
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(img, (cx, cy), radius, color, thickness, lineType=cv2.LINE_AA)
        if with_index:
            cv2.putText(img, str(i), (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)


class PlanVisualizer:
    """
    사용:
        viz = PlanVisualizer(cams, plan_path="...", show_window=True, video_path=None)
        canvas, quit_flag = viz.render(pkt)  # pkt: {"round", "coords", "fused", "timestamps"}
        viz.close()
    """
    def __init__(
        self,
        cams: List[Any],
        plan_path: Optional[str] = None,
        show_window: bool = True,
        window_name: str = "Plan Fusion",
        draw_cam_points: bool = False,
        cam_colors: Optional[List[Tuple[int, int, int]]] = None,
        fused_color: Tuple[int, int, int] = DEFAULT_FUSED_COLOR,
        video_path: Optional[str] = None,
        video_fps: int = 10,
        fallback_size=(2560, 1440),
    ):
        self.window_name = window_name
        self.show_window = show_window
        self.draw_cam_points = draw_cam_points
        self.cam_colors = cam_colors or DEFAULT_CAM_COLORS
        self.fused_color = fused_color

        base_img = get_plan_image_from_cams(cams)
        self.base = load_or_create_plan(base_img, plan_path, fallback_size=fallback_size)
        self.H, self.W = self.base.shape[:2]

        self.writer = None
        if video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(video_path, fourcc, video_fps, (self.W, self.H))

        if self.show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, min(self.W, 1280), min(self.H, 720))

    def render(self, pkt: dict, draw_cam_points: Optional[bool] = None):
        """
        한 라운드를 그려 반환.
        return: (canvas: np.ndarray, quit_flag: bool)
        """
        if draw_cam_points is None:
            draw_cam_points = self.draw_cam_points

        canvas = self.base.copy()

        # 카메라별 원시 좌표
        coords_per_cam = pkt.get("coords", [])
        if draw_cam_points and isinstance(coords_per_cam, list):
            for ci, cam_pts in enumerate(coords_per_cam):
                color = self.cam_colors[ci % len(self.cam_colors)]
                draw_points(canvas, cam_pts, color=color, radius=4, thickness=-1, with_index=False)

        # 정합된 좌표 (fused)
        fused = pkt.get("fused", []) or []
        draw_points(canvas, fused, color=self.fused_color, radius=6, thickness=-1, with_index=True, text_color=(255, 255, 255))

        # 텍스트 오버레이
        round_idx = pkt.get("round", -1)
        txt = f"Round {round_idx} | Fused: {len(fused)}"
        cv2.putText(canvas, txt, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (10, 10, 10), 2, cv2.LINE_AA)
        cv2.putText(canvas, txt, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 1, cv2.LINE_AA)

        # 디스플레이 / 저장
        quit_flag = False
        if self.show_window:
            cv2.imshow(self.window_name, canvas)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                quit_flag = True

        if self.writer is not None:
            self.writer.write(canvas)

        return canvas, quit_flag

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.show_window:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
