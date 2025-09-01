# /workspace/__Tracking/utils/video_decoder.py
import cv2
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    start: int
    end: int

def _cpu_decode_worker(args):
    path, chunk = args
    out = []
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
        out.append((fidx, frame))
    cap.release()
    return out

def iter_frames_parallel(video_path: str, cpu_workers: int = 8, chunk_sec: float = 10.0):
    cap0 = cv2.VideoCapture(video_path)
    if not cap0.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()

    if total <= 0:
        cap = cv2.VideoCapture(video_path)
        idx = -1
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            idx += 1
            yield idx, frame
        cap.release()
        return

    chunk_len = max(1, int(round(chunk_sec * fps)))
    chunks: List[Chunk] = []
    s = 0
    while s < total:
        e = min(total - 1, s + chunk_len - 1)
        chunks.append(Chunk(start=s, end=e))
        s = e + 1

    n_workers = min(max(1, int(cpu_workers)), len(chunks))
    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        for out in pool.imap(_cpu_decode_worker, [(video_path, c) for c in chunks]):
            out.sort(key=lambda x: x[0])
            for i, f in out:
                yield i, f
