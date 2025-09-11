# /workspace/__Tracking/utils/video_decoder.py
from __future__ import annotations
import os, threading, queue
from typing import Iterable, Tuple, Optional
import numpy as np

# PyAV가 있으면 사용, 없으면 OpenCV로 폴백
_USE_PYAV = True
try:
    import av  # PyAV
except Exception:
    _USE_PYAV = False
import cv2


def _bgr_from_avframe(frame) -> np.ndarray:
    """PyAV VideoFrame -> BGR ndarray"""
    return frame.to_ndarray(format="bgr24")


def iter_frames_streaming(
    video_path: str,
    *,
    target_size: Optional[Tuple[int, int]] = None,  # (width, height)
    decode_threads: int = 8,                        # FFmpeg/PyAV 디코더 스레드 (0=auto)
    prefetch_frames: int = 256,                     # 프리패치 큐 크기
    hwaccel: Optional[str] = None,                  # 'cuda' / 'nvdec' / None
    log_prefix: str = "[decoder]",
) -> Iterable[Tuple[int, np.ndarray]]:
    """
    순차 프레임 스트리밍 디코더.
    - 가능하면 PyAV(+FFmpeg) 사용, 실패 시 OpenCV 폴백
    - 출력: (frame_idx, BGR ndarray)
    - 순서 보장, 재정렬 불필요
    """
    assert os.path.exists(video_path), f"Missing video: {video_path}"
    q: "queue.Queue[Tuple[int, Optional[np.ndarray]]]" = queue.Queue(maxsize=prefetch_frames)

    def _worker_pyav():
        nonlocal q
        try:
            # FFmpeg 옵션 구성
            av_opts = {}
            if decode_threads is not None and decode_threads >= 0:
                # '0'이면 FFmpeg이 자동으로 결정
                av_opts["threads"] = "0" if decode_threads == 0 else str(decode_threads)

            # HW 가속 옵션 (NVDEC)
            hw = (hwaccel or "").lower() if hwaccel else ""
            if hw in ("cuda", "nvdec", "cuvid"):
                # 현재 FFmpeg는 'cuda' hwaccel을 권장
                av_opts["hwaccel"] = "cuda"
                av_opts["hwaccel_device"] = "0"  # 필요하면 변경
                # av_opts["hwaccel_output_format"] = "cuda"  # GPU frame 유지가 필요할 때만

            # 컨테이너 오픈
            container = av.open(video_path, mode="r", options=av_opts)
            # 비디오 스트림 선택
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                raise RuntimeError("No video stream found")

            # 코덱 내부 스레딩 힌트
            try:
                stream.thread_type = "AUTO"
                if decode_threads and decode_threads > 0:
                    stream.thread_count = int(decode_threads)
            except Exception:
                pass

            print(f"{log_prefix} open(PyAV): threads={decode_threads}, hwaccel={hw if hw else 'none'}")

            idx = 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    img = _bgr_from_avframe(frame)
                    if target_size:
                        w, h = target_size
                        if img.shape[1] != w or img.shape[0] != h:
                            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    q.put((idx, img))  # 소비자가 느리면 여기서 대기 (드롭 없음)
                    idx += 1
            container.close()
        except Exception as e:
            print(f"{log_prefix} PyAV error: {e} → fallback to OpenCV")
            # 폴백 실행
            _worker_cv2()
        finally:
            q.put((-1, None))  # sentinel

    def _worker_cv2():
        nonlocal q
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"{log_prefix} OpenCV cannot open: {video_path}")
            q.put((-1, None))
            return

        # OpenCV 내부 스레드 수 축소 (컨텍스트 스위칭 감소 목적)
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

        print(f"{log_prefix} open(OpenCV)")
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if target_size:
                w, h = target_size
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            q.put((idx, frame))
            idx += 1
        cap.release()

    # 백엔드 선택 및 스레드 시작
    if _USE_PYAV:
        t = threading.Thread(target=_worker_pyav, daemon=True)
    else:
        t = threading.Thread(target=_worker_cv2, daemon=True)
    t.start()

    # 소비자: 순차적으로 프레임 배출
    while True:
        idx, frame = q.get()
        if idx == -1 and frame is None:
            break
        yield idx, frame
