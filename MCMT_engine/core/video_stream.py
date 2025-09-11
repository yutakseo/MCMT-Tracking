# /workspace/MCMT_engine/core/video_stream.py
from __future__ import annotations
import os, threading, queue
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Callable
import numpy as np
import cv2

# PyAV가 있으면 우선 사용, 없으면 OpenCV 폴백
_HAVE_PYAV = True
try:
    import av  # PyAV
except Exception:
    _HAVE_PYAV = False


@dataclass
class VideoStreamConfig:
    target_size: Optional[Tuple[int, int]] = None   # (width, height)
    decode_threads: int = 0                         # FFmpeg 내부 디코드 스레드; 0=auto
    prefetch_frames: int = 256                      # 프리패치 큐 크기
    hwaccel: Optional[str] = None                   # 'cuda' or None
    log_prefix: str = "[decoder]"


def _bgr_from_avframe(frame) -> np.ndarray:
    # PyAV VideoFrame -> BGR ndarray (스케일/색변환 포함)
    return frame.to_ndarray(format="bgr24")


def iter_frames_streaming(video_path: str, cfg: VideoStreamConfig) -> Iterable[Tuple[int, np.ndarray]]:
    """
    순차 프레임 스트리밍 (프레임 인덱스, BGR ndarray) 생성기.
    - PyAV(+FFmpeg) 우선, 실패 시 OpenCV로 폴백
    - 순서 보장, 재정렬 불필요
    """
    assert os.path.exists(video_path), f"Missing video: {video_path}"
    q: "queue.Queue[Tuple[int, Optional[np.ndarray]]]" = queue.Queue(maxsize=cfg.prefetch_frames)

    def _worker_pyav():
        nonlocal q
        try:
            # av.open 옵션 구성
            open_opts = {}
            if decode_threads is not None and decode_threads >= 0:
                # '0'은 FFmpeg의 자동 결정(= 코덱 내부 스레드 최대 활용)
                open_opts["threads"] = "0" if decode_threads == 0 else str(decode_threads)

            # NVDEC 힌트 (가능할 때만)
            if hwaccel == "cuda":
                # hwaccel_device는 환경에 맞춰 GPU 인덱스 지정(여기선 0)
                open_opts["hwaccel"] = "cuda"
                open_opts["hwaccel_device"] = "0"
                # 필요시 아래 옵션도 시도해볼 수 있음 (환경 따라 무시되기도 함)
                # open_opts["hwaccel_output_format"] = "cuda"

            print(f"{log_prefix} open(PyAV): threads={decode_threads}, hwaccel={hwaccel}")
            container = av.open(video_path, mode="r", options=open_opts)

            # 첫 번째 비디오 스트림 선택
            stream = next(s for s in container.streams if s.type == "video")

            # 코덱 내부 스레딩 힌트
            try:
                stream.thread_type = "AUTO"
                if decode_threads and decode_threads > 0:
                    stream.thread_count = int(decode_threads)
            except Exception:
                pass

            idx = 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    # 주의: 여기서 to_ndarray('bgr24')를 하면 CPU 메모리로 내려옵니다.
                    # 그래도 "디코딩" 자체는 NVDEC이 처리하므로 CPU 부하는 줄고
                    # nvidia-smi에서 decoder%가 올라갑니다.
                    img = _bgr_from_avframe(frame)
                    if target_size:
                        w, h = target_size
                        if img.shape[1] != w or img.shape[0] != h:
                            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
                    q.put((idx, img))
                    idx += 1

            container.close()
        except Exception as e:
            print(f"{log_prefix} PyAV error: {e}")
        finally:
            q.put((-1, None))  # sentinel


    def _worker_cv2():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"{cfg.log_prefix} OpenCV cannot open: {video_path}")
            q.put((-1, None))
            return
        try:
            # OpenCV 내부 스레드 줄여서 컨텍스트 스위칭 비용 감소
            cv2.setNumThreads(1)
        except Exception:
            pass

        idx = 0
        print(f"{cfg.log_prefix} open(OpenCV): target_size={cfg.target_size}")
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if cfg.target_size:
                w, h = cfg.target_size
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            q.put((idx, frame))
            idx += 1
        cap.release()
        q.put((-1, None))

    t = threading.Thread(target=_worker_pyav if _HAVE_PYAV else _worker_cv2, daemon=True)
    t.start()

    while True:
        idx, frame = q.get()
        if idx == -1 and frame is None:
            break
        yield idx, frame


# ---- DI(의존성 주입)용 팩토리 ----
def stream_factory_from_args(args) -> Callable[[str], Iterable[Tuple[int, np.ndarray]]]:
    """
    Args 객체로부터 스트리밍 설정을 생성해, (video_path) -> iterator 호출 가능한
    streamer 함수를 반환.
    """
    cfg = VideoStreamConfig(
        target_size=getattr(args, "decode_target_size", None),
        decode_threads=int(
            getattr(args, "decode_threads",
                    getattr(args, "cpu_workers", 0)) or 0
        ),
        prefetch_frames=int(getattr(args, "prefetch_frames", 256)),
        hwaccel=getattr(args, "hwaccel", None),
        log_prefix="[decoder]",
    )

    def _streamer(video_path: str):
        return iter_frames_streaming(video_path, cfg)

    # 사용자가 로그에서 현재 설정을 보려면 cfg를 노출해도 됨
    _streamer.cfg = cfg  # type: ignore[attr-defined]
    return _streamer


# ---- 구(legacy) 시그니처 어댑터 (원하면 다른 곳에서 재사용 가능) ----
def iter_frames_streaming_legacy(
    video_path: str,
    *,
    target_size: Optional[Tuple[int, int]] = None,
    decode_threads: int = 0,
    prefetch_frames: int = 256,
    hwaccel: Optional[str] = None,
    log_prefix: str = "[decoder]",
) -> Iterable[Tuple[int, np.ndarray]]:
    cfg = VideoStreamConfig(
        target_size=target_size,
        decode_threads=decode_threads,
        prefetch_frames=prefetch_frames,
        hwaccel=hwaccel,
        log_prefix=log_prefix,
    )
    return iter_frames_streaming(video_path, cfg)
