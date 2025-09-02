# /workspace/tools/webviz.py
from __future__ import annotations

import io
import time
from typing import List, Tuple, Optional, Dict, Any
import threading

import cv2
import numpy as np


class WebPlanViz:
    """
    - plan_path 또는 plan_img 중 하나 제공.
    - update(pkt)로 최신 프레임 생성 (pkt는 {"fused": [(x,y), ...], "coords": [[(x,y),...], ...]} 포맷 가정)
    - mjpeg_generator()를 FastAPI StreamingResponse에 넘겨 웹으로 MJPEG 스트림 제공.
    """

    def __init__(
        self,
        plan_path: Optional[str] = None,
        plan_img: Optional[np.ndarray] = None,
        *,
        show_cam_points: bool = False,   # 카메라별 점도 보이려면 True
        dot_radius: int = 6,
        dot_thickness: int = -1,         # 채우기 (-1)
        fused_color: Tuple[int, int, int] = (0, 255, 0),
        cam_colors: Optional[List[Tuple[int, int, int]]] = None,
        text_color: Tuple[int, int, int] = (255, 255, 255),
        font_scale: float = 0.6,
        fps_limit: float = 20.0,         # 스트림 최대 FPS
    ):
        # 베이스 플랜 이미지 준비
        base = None
        if plan_img is not None and isinstance(plan_img, np.ndarray) and plan_img.ndim >= 2:
            base = plan_img.copy()
        elif plan_path is not None:
            img = cv2.imread(plan_path, cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Failed to read plan image: {plan_path}")
            base = img
        else:
            # fallback: 검정 바탕 1080p
            base = np.zeros((1080, 1920, 3), dtype=np.uint8)

        if base.ndim == 2:
            base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        self._base = base
        self._H, self._W = base.shape[:2]

        self.show_cam_points = show_cam_points
        self.dot_radius = int(dot_radius)
        self.dot_thickness = int(dot_thickness)
        self.fused_color = fused_color
        self.cam_colors = cam_colors or [
            (255, 200, 0),   # BGR (카메라 1)
            (0, 200, 255),   # 카메라 2
            (180, 0, 255),   # 카메라 3
            (0, 255, 200),   # 카메라 4
            (200, 255, 0),   # 카메라 5
            (255, 0, 120),   # 카메라 6
        ]
        self.text_color = text_color
        self.font_scale = float(font_scale)
        self.fps_limit = float(fps_limit)

        # 스트림 공유 상태
        self._lock = threading.Lock()
        self._last_jpeg: bytes = self._encode_jpeg(self._base)  # 초기 화면
        self._last_render_t: float = 0.0
        self._min_interval = 0.0 if self.fps_limit <= 0 else (1.0 / self.fps_limit)

    # ──────────────────────────────────────────────────────────────────────
    # 외부 API
    # ──────────────────────────────────────────────────────────────────────
    def update(self, pkt: Dict[str, Any]) -> None:
        """
        pkt: {"fused": [(x,y), ...], "coords": [[(x,y), ...], ...], "round": int(optional)}
        """
        fused = pkt.get("fused") or []
        coords_per_cam = pkt.get("coords") or []

        # FPS 제한
        now = time.time()
        if now - self._last_render_t < self._min_interval:
            # 그래도 최신 프레임으로 한 번 갱신은 해주자 (너무 과하면 drop)
            pass
        self._last_render_t = now

        frame = self._draw(fused, coords_per_cam, round_idx=pkt.get("round"))
        jpg = self._encode_jpeg(frame)
        with self._lock:
            self._last_jpeg = jpg

    def mjpeg_generator(self):
        """
        FastAPI StreamingResponse에 바인딩할 제너레이터.
        최신 JPEG을 boundary 포맷으로 계속 내보냄.
        """
        boundary = b"--frame"
        while True:
            with self._lock:
                jpg = self._last_jpeg
            # multipart/x-mixed-replace; boundary=frame
            yield boundary + b"\r\n"
            yield b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
            yield jpg + b"\r\n"
            # 약간의 휴지 (너무 과도한 송신 방지)
            if self._min_interval > 0:
                time.sleep(self._min_interval)

    # ──────────────────────────────────────────────────────────────────────
    # 내부: 렌더링
    # ──────────────────────────────────────────────────────────────────────
    def _draw(
        self,
        fused: List[Tuple[float, float]],
        coords_per_cam: List[List[Tuple[float, float]]],
        round_idx: Optional[int] = None,
    ) -> np.ndarray:
        canvas = self._base.copy()

        # 카메라별 포인트
        if self.show_cam_points and coords_per_cam:
            for ci, pts in enumerate(coords_per_cam):
                col = self.cam_colors[ci % len(self.cam_colors)]
                for (x, y) in pts:
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= xi < self._W and 0 <= yi < self._H:
                        cv2.circle(canvas, (xi, yi), max(2, self.dot_radius - 2), col, self.dot_thickness)

        # Fused 포인트 (굵게)
        for (x, y) in fused:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < self._W and 0 <= yi < self._H:
                cv2.circle(canvas, (xi, yi), self.dot_radius, self.fused_color, self.dot_thickness)

        # 헤더 텍스트
        header = f"Fused: {len(fused)}"
        if round_idx is not None:
            header += f" | Round: {round_idx}"
        cv2.putText(
            canvas, header, (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, 2, cv2.LINE_AA
        )
        return canvas

    @staticmethod
    def _encode_jpeg(img: np.ndarray, quality: int = 80) -> bytes:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            # 드물게 인코딩 실패하면 대체
            h, w = img.shape[:2]
            fallback = np.zeros((max(h, 1), max(w, 1), 3), dtype=np.uint8)
            ok, buf = cv2.imencode(".jpg", fallback, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return buf.tobytes()
