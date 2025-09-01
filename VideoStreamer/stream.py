# stream_cctv.py
import cv2
import threading
import time
from typing import Optional

class StreamCCTV:
    """
    CCTV(RTSP/HTTP) 스트리밍을 백그라운드에서 계속 읽고,
    capture() 호출 시 최신 프레임 한 장을 반환.
    """

    def __init__(self, url: str, max_width: int = 0, reconnect_delay: float = 2.0):
        """
        Args:
            url: CCTV 스트리밍 URL
            max_width: 리사이즈 최대 가로폭 (0이면 원본 유지)
            reconnect_delay: 연결 실패시 재시도 간격 (초)
        """
        self.url = url
        self.max_width = max_width
        self.reconnect_delay = reconnect_delay

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[cv2.Mat] = None
        self._ts: float = 0.0

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---------------------------
    # Public API
    # ---------------------------
    def start(self):
        """스트리밍 시작 (백그라운드 루프 실행)"""
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """스트리밍 종료"""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._release_cap()

    def capture(self, copy: bool = False):
        """
        최신 프레임 한 장 반환.
        - copy=False: 내부 버퍼 참조 반환 (빠름, 읽기 전용 권장)
        - copy=True : 프레임 복사 후 반환 (수정할 경우 안전)
        """
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy() if copy else self._frame

    def last_timestamp(self) -> float:
        """마지막 프레임 수신 시각"""
        return self._ts

    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ---------------------------
    # 내부 로직
    # ---------------------------
    def _open_cap(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _release_cap(self):
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    def _resize_if_needed(self, frame):
        if self.max_width and frame is not None:
            h, w = frame.shape[:2]
            if w > self.max_width:
                scale = self.max_width / float(w)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        return frame

    def _update(self):
        """백그라운드에서 무한히 스트리밍"""
        while not self._stop.is_set():
            if self._cap is None:
                self._cap = self._open_cap()
                if self._cap is None:
                    time.sleep(self.reconnect_delay)
                    continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                # 재연결
                self._release_cap()
                time.sleep(self.reconnect_delay)
                continue

            frame = self._resize_if_needed(frame)
            with self._lock:
                self._frame = frame
                self._ts = time.time()

        self._release_cap()
