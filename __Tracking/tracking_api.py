# /workspace/__Tracking/tracking_api.py
from __Tracking.ByteTrack.yolox.tracker.byte_tracker import BYTETracker
import cv2, os
import numpy as np
from hashlib import md5
from collections import defaultdict, deque
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import multiprocessing as mp

# =========================
# 병렬 디코딩 유틸
# =========================
@dataclass
class _Chunk:
    start: int  # inclusive
    end: int    # inclusive

def _cpu_decode_worker(args):
    """
    하나의 연속 구간[start..end]을 디코딩해 (idx, frame[BGR]) 리스트로 반환.
    """
    path, chunk = args
    out = []
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return out
    # 시킹 (코덱에 따라 정확도 차이)
    cap.set(cv2.CAP_PROP_POS_FRAMES, chunk.start)
    fidx = chunk.start - 1
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        fidx += 1
        if fidx > chunk.end:
            break
        out.append((fidx, frame))  # BGR
    cap.release()
    return out

def _iter_frames_parallel(video_path: str,
                          cpu_workers: int = 8,
                          chunk_sec: float = 20.0):
    """
    전체 프레임을 (idx, frame[BGR]) 순서로 스트리밍.
    - 청크 단위로 병렬 디코딩하되,
    - imap(순서 보장)으로 '글로벌 프레임 순서'를 그대로 유지해서 yield.
    """
    cap0 = cv2.VideoCapture(video_path)
    if not cap0.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap0.release()

    if total <= 0:
        # 폴백: 단일 프로세스 순차 디코딩
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
        return

    # 청크 분할
    chunk_len = max(1, int(round(chunk_sec * fps)))
    chunks: List[_Chunk] = []
    s = 0
    while s < total:
        e = min(total - 1, s + chunk_len - 1)
        chunks.append(_Chunk(start=s, end=e))
        s = e + 1

    n_workers = max(1, int(cpu_workers))
    n_workers = min(n_workers, len(chunks))

    # 순서 보장 imap 사용 → 청크 순서 == 프레임 글로벌 순서
    with mp.get_context("spawn").Pool(processes=n_workers) as pool:
        for out in pool.imap(_cpu_decode_worker, [(video_path, c) for c in chunks]):
            # 해당 청크 내부 정렬(안전)
            out.sort(key=lambda x: x[0])
            for i, f in out:
                yield i, f

# =========================
# 원본 API (추론/추적은 그대로)
# =========================
class TrackerAPI:
    def __init__(self, args, detector) -> None:
        """
        args: tracker 설정값 객체 (track_thresh, match_thresh, track_buffer 등) ← BYTETracker용
              추가로 있으면 좋음 (없으면 기본값 사용):
                - cpu_workers: int, 병렬 디코딩 워커 수 (기본 8)
                - chunk_sec: float, 청크 길이 초 (기본 20.0)
        detector: .detect(image) -> (N,5 or N,6) 텐서/넘파이 (x1,y1,x2,y2,score,(cls))
        """
        self.tracker = BYTETracker(args)
        self.detector = detector

        # 디코딩 병렬화 설정 (없으면 기본값)
        self.cpu_workers = int(getattr(args, "cpu_workers", 8))
        self.chunk_sec   = float(getattr(args, "chunk_sec", 20.0))

    def tracking(self, video_path, visualize=False):
        """
        video_path: 비디오 파일 경로 (ex: "video.mp4")
        visualize: True일 경우 결과를 화면에 그려줌

        변경점: cv2.VideoCapture 반복 대신, 병렬 디코딩 이터레이터로 프레임 공급.
        추론/추적 로직은 기존과 동일하게 '프레임별 순차' 처리.
        """
        results = []
        img_size = None

        # 병렬 디코딩 이터레이터 (프레임 글로벌 순서 보장)
        frame_stream = _iter_frames_parallel(
            video_path,
            cpu_workers=self.cpu_workers,
            chunk_sec=self.chunk_sec
        )

        for _, frame in frame_stream:
            if frame is None:
                continue

            # 첫 프레임에서 img_size 자동 결정
            if img_size is None:
                fh, fw = frame.shape[:2]
                img_size = (fh, fw)

            # 1) Detection (프레임별)
            dets = self.detector.detect(frame)

            # 2) Tracking (그대로)
            fh, fw = frame.shape[:2]
            info_imgs = (fh, fw)
            online_targets = self.tracker.update(dets, info_imgs, img_size)

            frame_res = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                score = t.score
                frame_res.append({
                    "id": tid,
                    "bbox": tlwh,   # (x, y, w, h) float32
                    "score": score
                })

                if visualize:
                    x1, y1, bw, bh = map(int, tlwh)
                    cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{tid}", (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            results.append(frame_res)

            if visualize:
                cv2.imshow("tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if visualize:
            cv2.destroyAllWindows()
        return results

    def _color_from_id(self, track_id: int):
        # ID마다 고정 색상
        h = md5(str(track_id).encode()).hexdigest()
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))  # BGR-ish

    def trackingVideo(self, video_path, save_path, trail_len: int = 30):
        """
        video_path: 입력 영상 경로
        save_path : 저장할 결과 영상 경로
        trail_len: 궤적 길이(저장할 최근 중심점 개수)

        주의: tracking()은 모든 프레임을 순서대로 처리하므로,
             여기서는 원본 비디오를 '그대로' 순차 읽으며 결과를 덧그린다.
        """
        # 1) detection+tracking (화면표시 X)
        results = self.tracking(video_path=video_path, visualize=False)

        # 2) 비디오 입출력 준비
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1e-3:
            fps = 30.0  # 기본값

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            alt_path = os.path.splitext(save_path)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(alt_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError("VideoWriter open failed for both mp4v and MJPG.")
            save_path = alt_path

        # 3) ID별 궤적 저장소
        trails = defaultdict(lambda: deque(maxlen=trail_len))
        trail_thickness = 2

        # 4) 프레임 순회하며 시각화 + 저장
        res_iter = iter(results)
        for frame_id in range(len(results)):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            try:
                frame_res = next(res_iter)
            except StopIteration:
                break

            current_ids = set()
            for t in frame_res:
                x, y, bw, bh = map(int, t["bbox"])
                tid = int(t["id"])
                score = float(t["score"])
                current_ids.add(tid)

                color = self._color_from_id(tid)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(frame, f"ID:{tid} {score:.2f}",
                            (x, max(0, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 중심점 갱신
                cx = x + bw * 0.5
                cy = y + bh * 0.5
                trails[tid].append((cx, cy))

            # 궤적 그리기
            for tid in current_ids:
                pts = trails[tid]
                if len(pts) >= 2:
                    pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts_np], isClosed=False,
                                  color=self._color_from_id(tid), thickness=trail_thickness)
                if len(pts) >= 1:
                    cx, cy = map(int, pts[-1])
                    cv2.circle(frame, (cx, cy), radius=2 + trail_thickness,
                               color=self._color_from_id(tid), thickness=-1)

            writer.write(frame)

        cap.release()
        writer.release()
        print(f"결과 영상 저장 완료: {save_path}")
        return results
