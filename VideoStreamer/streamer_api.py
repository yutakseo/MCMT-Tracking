# streamer_api.py
# --- 런타임 로더/링커 이슈 방지: 시스템 libstdc++ 선로딩 + OpenCV so 경로 우선 ---
import os, sys, ctypes, time

# 1) 시스템 libstdc++ 를 전역으로 선로딩 (GDAL이 요구하는 GLIBCXX_3.4.30 충족)
SYS_LIBSTDCPP = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.path.exists(SYS_LIBSTDCPP):
    try:
        ctypes.CDLL(SYS_LIBSTDCPP, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        # 선로딩 실패해도 이후 import에서 다시 시도되므로 묵살
        pass

# 2) OpenCV .so 들이 있는 시스템 경로를 우선 추가
SYS_LIB_DIR = "/usr/lib/x86_64-linux-gnu"
os.environ["LD_LIBRARY_PATH"] = f"{SYS_LIB_DIR}:" + os.environ.get("LD_LIBRARY_PATH", "")

# 3) C++ 확장(.so) 모듈이 있는 build 경로를 sys.path 맨 앞에
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "build"))

import cv2
import numpy as np
import stream_cctv_cpp


class CCTVStreamer:
    """
    C++ 기반 StreamCCTV(pybind11) 를 파이썬에서 안전/간편하게 쓰기 위한 래퍼.

    - start()가 self를 반환하므로, CCTVStreamer(...).start() 체이닝 가능
    - capture(copy=False) 기본: zero-copy로 빠름. 프레임 수정 필요할 때만 copy=True
    - show()는 헤드리스 환경에서도 예외 없이 동작하도록 예외 처리 포함
    - wait_ready() 제공: 첫 프레임 도착 대기
    """
    def __init__(self, url: str, max_width: int = 960, window_name: str = "CCTV", reconnect_delay: int = 2):
        self.url = url
        self.max_width = int(max_width)
        self.window_name = window_name
        self._reconnect_delay = int(reconnect_delay)
        self.cam = stream_cctv_cpp.StreamCCTV(self.url, self.max_width, self._reconnect_delay)
        self._started = False

    # --- Context manager 지원 ---
    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()

    # --- Control ---
    def start(self):
        """스트리밍 시작 (중복 시작 방지). 체이닝을 위해 self 반환."""
        if not self._started:
            self.cam.start()
            self._started = True
        return self

    def stop(self):
        """스트리밍 정지 및 창 정리(헤드리스 안전)."""
        if self._started:
            try:
                self.cam.stop()
            finally:
                self._started = False
        # GUI 환경이 아닐 수 있어 예외 안전
        try:
            cv2.destroyWindow(self.window_name)
        except cv2.error:
            pass
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass

    def is_started(self) -> bool:
        return self._started

    # --- IO ---
    def capture(self, copy: bool = False):
        """
        최신 프레임을 numpy.ndarray(BGR, HxWx3)로 반환. 없으면 None.
        copy=True면 내부 버퍼와 분리된 독립 메모리를 반환.
        """
        frame = self.cam.capture(copy=copy)
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        # pybind11에서 이미 (H, W, C) 형태로 넘어옴
        return frame

    def wait_ready(self, timeout: float = 2.0, poll: float = 0.01) -> bool:
        """
        첫 유효 프레임이 들어올 때까지 대기.
        True: 프레임 확보 / False: 타임아웃
        """
        end = time.time() + float(timeout)
        while time.time() < end:
            f = self.capture()
            if f is not None:
                return True
            time.sleep(poll)
        return False

    def show(self, window_name: str = "CCTV", delay: int = 1, quit_key: str = "q", wait_ready_timeout: float = 0.0):
        """
        프레임을 윈도우에 띄움. quit_key 입력 시 종료.
        - wait_ready_timeout > 0 이면 첫 프레임 대기 후 시작
        """
        self.window_name = window_name
        self.start()

        # 첫 프레임 대기 옵션
        if wait_ready_timeout and wait_ready_timeout > 0:
            self.wait_ready(timeout=wait_ready_timeout)

        # 창 생성(헤드리스 환경이면 아래 imshow에서 예외 처리)
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except cv2.error:
            pass

        try:
            while True:
                frame = self.capture()
                if frame is not None:
                    try:
                        cv2.imshow(self.window_name, frame)
                    except cv2.error:
                        # 헤드리스 환경: 그냥 프레임 소비하고 계속
                        pass
                else:
                    time.sleep(0.005)
                if cv2.waitKey(delay) & 0xFF == ord(quit_key):
                    break
        finally:
            self.stop()


