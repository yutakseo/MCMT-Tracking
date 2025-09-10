# /workspace/__Tracking/utils/video_decoder.py
# -*- coding: utf-8 -*-
"""
안정형 병렬 비디오 디코더
- 플랫폼별 멀티프로세싱 start method 자동 선택 (Linux: fork, Windows: spawn)
- spawn 문제(메인가드 누락/디버거/Jupyter 등) 발생 시 순차 모드로 안전 폴백
- Pool 종료 시점 명시(terminate/close/join) + maxtasksperchild=1로 OpenCV 핸들 누수 완화
- KeyboardInterrupt 안전 처리
- 총 프레임 수를 얻지 못하는 영상은 자동으로 순차 디코딩
"""

from dataclasses import dataclass
from typing import List, Iterator, Tuple, Optional

import cv2
import numpy as np
import multiprocessing as mp
import warnings
import sys


@dataclass
class Chunk:
    start: int
    end: int


def _cpu_decode_worker(args: Tuple[str, Chunk]) -> List[Tuple[int, np.ndarray]]:
    """단일 청크를 디코드하여 (frame_index, frame_bgr) 리스트로 반환."""
    path, chunk = args

    # OpenCV 내부 스레드 과다 생성 방지(환경 따라 무시될 수 있음)
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    out: List[Tuple[int, np.ndarray]] = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return out

    cap.set(cv2.CAP_PROP_POS_FRAMES, chunk.start)
    fidx = chunk.start - 1

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        fidx += 1
        if fidx > chunk.end:
            break
        # 프레임 복사 불필요(성능상 이점); 필요 시 frame.copy() 사용
        out.append((fidx, frame))

    cap.release()
    return out


def _iter_frames_sequential(video_path: str) -> Iterator[Tuple[int, np.ndarray]]:
    """순차 디코딩 제너레이터."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    idx = -1
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        idx += 1
        yield idx, frame

    cap.release()


def _build_chunks(total_frames: int, chunk_len: int) -> List[Chunk]:
    """총 프레임 수와 청크 길이로 연속 청크 리스트 작성."""
    chunks: List[Chunk] = []
    s = 0
    while s < total_frames:
        e = min(total_frames - 1, s + chunk_len - 1)
        chunks.append(Chunk(start=s, end=e))
        s = e + 1
    return chunks


def iter_frames_parallel(
    video_path: str,
    cpu_workers: int = 8,
    chunk_sec: float = 10.0,
    start_method: Optional[str] = None,  # 'fork' | 'spawn' | None(자동)
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    병렬 디코딩 제너레이터.
    - cpu_workers <= 1 이면 자동으로 순차 디코딩으로 전환.
    - 총 프레임 수를 알 수 없거나 풀 생성 실패/예외 발생 시 순차로 폴백.
    - 결과는 (frame_index, frame_bgr) 튜플로 출력.
      (청크 내부는 정렬, 청크 간 순서는 보장하지 않음 → 상위에서 index 기반 재정렬 가능)
    """
    # 워커 1 이하면 순차
    if cpu_workers <= 1:
        yield from _iter_frames_sequential(video_path)
        return

    # 기본 메타 정보 파싱
    cap0 = cv2.VideoCapture(video_path)
    if not cap0.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap0.get(cv2.CAP_PROP_FPS)
    if fps is None or np.isnan(fps) or fps < 1e-3:
        fps = 30.0

    total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()

    # 총 프레임 수를 얻지 못하면 순차
    if total <= 0:
        warnings.warn("[iter_frames_parallel] total frames unknown → sequential fallback")
        yield from _iter_frames_sequential(video_path)
        return

    # 청크 구성
    chunk_len = max(1, int(round(chunk_sec * fps)))
    chunks = _build_chunks(total_frames=total, chunk_len=chunk_len)

    # 워커 수 결정
    n_workers = min(max(1, int(cpu_workers)), len(chunks))

    # 플랫폼별 start method 자동 선택
    if start_method is None:
        start_method = "fork" if sys.platform != "win32" else "spawn"

    # Pool 생성/실행
    try:
        ctx = mp.get_context(start_method)
        pool = ctx.Pool(processes=n_workers, maxtasksperchild=1)  # 누수/핸들 문제 완화

        try:
            # unordered가 보통 더 빠름(상위에서 index로 재정렬 가능)
            for out in pool.imap_unordered(_cpu_decode_worker, [(video_path, c) for c in chunks], chunksize=1):
                if not out:
                    continue
                # 청크 내부 정렬
                out.sort(key=lambda x: x[0])
                for i, f in out:
                    yield i, f
        except KeyboardInterrupt:
            # 사용자 중단 시 깔끔히 정리
            pool.terminate()
            pool.join()
            raise
        except Exception:
            # 예외 시 바로 종료
            pool.terminate()
            pool.join()
            raise
        else:
            # 정상 종료
            pool.close()
            pool.join()

    except RuntimeError as e:
        # spawn 관련 에러(메인가드 누락 등) 또는 기타 컨텍스트 생성 실패 → 순차 폴백
        warnings.warn(f"[iter_frames_parallel] parallel failed ({e}). Falling back to sequential.")
        yield from _iter_frames_sequential(video_path)
