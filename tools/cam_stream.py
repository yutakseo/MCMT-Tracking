# /workspace/tools/cam_stream.py
from __future__ import annotations
import cv2, threading, time
import numpy as np
from typing import Optional, Union

# CCTVStreamer가 있으면 우선 사용
try:
    from VideoStreamer.streamer_api import CCTVStreamer  # <- 당신이 준 모듈
    _HAS_CCTV = True
except Exception:
    CCTVStreamer = None  # type: ignore
    _HAS_CCTV = False


class CamMJPEG:
    """
    RTSP → MJPEG 스트리머.
    - CCTVStreamer(pybind11) 우선 사용, 없으면 OpenCV VideoCapture 폴백
    - start().mjpeg_generator()를 FastAPI StreamingResponse로 연결
    """
    def __init__(
        self,
        name: str,
        url: Optional[str] = None,
        streamer: Optional["CCTVStreamer"] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        overlay: bool = True,
        jpeg_quality: int = 80,
    ):
        assert (url is not None) or (streamer is not None), "url 또는 streamer 둘 중 하나는 필요합니다."
        self.name = name
        self.url = url
        self._user_streamer = streamer
        self.width = width
        self.height = height
        self.overlay = overlay
        self.jpeg_quality = int(np.clip(jpeg_quality, 40, 95))

        # 내부 상태
        self._lock = threading.Lock()
        self._jpg: Optional[bytes] = self._make_placeholder(f"Waiting: {self.name}")
        self._stop = False
        self._th: Optional[threading.Thread] = None

        # 백엔드 선택
        self._backend = "cctv" if (_HAS_CCTV and (streamer is not None or url is not None)) else "opencv"
        self._cap = None
        self._cctv: Optional[CCTVStreamer] = None

    def start(self) -> "CamMJPEG":
        if self._th and self._th.is_alive():
            return self
        self._stop = False

        # 백엔드 초기화
        if self._backend == "cctv":
            if self._user_streamer is not None:
                self._cctv = self._user_streamer
            else:
                # CCTVStreamer(url, max_width=...) 를 생성
                mw = self.width if self.width else 960
                self._cctv = CCTVStreamer(self.url, max_width=int(mw), window_name=self.name)  # type: ignore
            self._cctv.start()  # type: ignore
        else:
            # OpenCV 백엔드
            self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)  # type: ignore

        self._th = threading.Thread(target=self._reader_loop, daemon=True)
        self._th.start()
        return self

    def stop(self):
        self._stop = True
        if self._th and self._th.is_alive():
            self._th.join(timeout=1.0)
        if self._backend == "cctv":
            try:
                self._cctv and self._cctv.stop()  # type: ignore
            except Exception:
                pass
        else:
            try:
                self._cap and self._cap.release()
            except Exception:
                pass

    def mjpeg_generator(self, fps: int = 20):
        boundary = b"--frame"
        delay = 1.0 / max(1, fps)
        while True:
            with self._lock:
                b = self._jpg
            if b is None:
                b = self._make_placeholder(f"No frame: {self.name}")
            yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + b + b"\r\n"
            time.sleep(delay)

    # ── 내부 루프 ─────────────────────────────────────────────────────────
    def _reader_loop(self):
        backoff = 0.5
        while not self._stop:
            try:
                while not self._stop:
                    frame = None
                    if self._backend == "cctv":
                        # pybind11 백엔드 — zero-copy라 copy=True 권장
                        frame = self._cctv.capture(copy=True) if self._cctv else None  # type: ignore
                    else:
                        if self._cap is None or not self._cap.isOpened():
                            raise RuntimeError("VideoCapture not opened")
                        ok, f = self._cap.read()
                        frame = f if ok else None

                    if frame is None:
                        # 입력 끊김: 재시도 / CCTVStreamer는 내부 재연결 로직이 있어 살짝 대기
                        time.sleep(0.02)
                        continue

                    # 리사이즈
                    if self.width or self.height:
                        w = self.width or frame.shape[1]
                        h = self.height or frame.shape[0]
                        if (w, h) != (frame.shape[1], frame.shape[0]):
                            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

                    if self.overlay:
                        ts = time.strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(frame, f"{self.name}  {ts}", (10, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                    if ok:
                        with self._lock:
                            self._jpg = jpg.tobytes()

                break  # stop
            except Exception:
                with self._lock:
                    self._jpg = self._make_placeholder(f"Reconnecting: {self.name}")
                time.sleep(backoff)
                backoff = min(5.0, backoff * 2.0)

    @staticmethod
    def _make_placeholder(text: str) -> bytes:
        img = np.zeros((240, 320, 3), np.uint8)
        img[:] = (16, 16, 16)
        cv2.putText(img, text, (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        ok, jpg = cv2.imencode(".jpg", img)
        return jpg.tobytes() if ok else b""
