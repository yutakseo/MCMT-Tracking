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
                          chunk_sec: float = 10.0):
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
# Tracker API
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
        self.args = args
        self.tracker = BYTETracker(args)
        self.detector = detector

        # 디코딩 병렬화 설정 (없으면 기본값)
        self.cpu_workers = int(getattr(args, "cpu_workers", 8))
        self.chunk_sec   = float(getattr(args, "chunk_sec", 20.0))

        # 기존 변수명과 동일하게 유지 (img_size 등)
        self.img_size: Optional[Tuple[int, int]] = None  # (fh, fw)

    # ----------------------------
    # 유틸/시각화 (분리된 메서드)
    # ----------------------------
    def _color_from_id(self, track_id: int):
        # ID마다 고정 색상
        h = md5(str(track_id).encode()).hexdigest()
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))  # BGR-ish

    def visualize_frame(
        self,
        frame: np.ndarray,
        frame_res: List[Dict[str, Any]],
        trails: Optional[Dict[int, deque]] = None,
        trail_len: int = 30,
        trail_thickness: int = 2,
        draw_score: bool = True,
        copy: bool = True,
    ) -> np.ndarray:
        """
        boxes/ID/점수/궤적 그리기 (변수명: frame, frame_res, trails, trail_thickness 유지)
        """
        vis = frame.copy() if copy else frame
        if trails is None:
            trails = {}

        current_ids = set()
        for t in frame_res:
            x, y, bw, bh = map(int, t["bbox"])
            tid = int(t["id"])
            score = float(t.get("score", 0.0))
            current_ids.add(tid)

            color = self._color_from_id(tid)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
            label = f"ID:{tid}" + (f" {score:.2f}" if draw_score else "")
            cv2.putText(vis, label, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if trail_len > 0:
                if tid not in trails:
                    trails[tid] = deque(maxlen=trail_len)
                cx = x + bw * 0.5
                cy = y + bh * 0.5
                trails[tid].append((cx, cy))

        # 궤적 그리기 (변수명 동일)
        if trail_len > 0:
            for tid in current_ids:
                pts = trails.get(tid, None)
                if pts and len(pts) >= 2:
                    pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(vis, [pts_np], isClosed=False,
                                  color=self._color_from_id(tid), thickness=trail_thickness)
                if pts and len(pts) >= 1:
                    cx, cy = map(int, pts[-1])
                    cv2.circle(vis, (cx, cy), radius=2 + trail_thickness,
                               color=self._color_from_id(tid), thickness=-1)
        return vis

    # ----------------------------
    # 이미지(단일 프레임) 추적
    # ----------------------------
    def track_image(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        단일 프레임(BGR) 추적.
        - 이 인스턴스를 계속 사용하면 ByteTrack 상태가 이어져 ID가 유지됨.
        return: frame_res = [{"id":int, "bbox":(x,y,w,h), "score":float}, ...]
        """
        if frame is None:
            return []

        fh, fw = frame.shape[:2]
        if self.img_size is None:
            self.img_size = (fh, fw)  # 첫 프레임에서 기준 고정

        # 1) Detection
        dets = self.detector.detect(frame)

        # 2) Tracking (기존 변수명 유지)
        info_imgs = (fh, fw)
        online_targets = self.tracker.update(dets, info_imgs, self.img_size)

        frame_res = []
        for t in online_targets:
            tlwh = t.tlwh  # (x, y, w, h)
            tid = t.track_id
            score = t.score
            frame_res.append({
                "id": int(tid),
                "bbox": tlwh,
                "score": float(score)
            })
        return frame_res

    # ----------------------------
    # 비디오 추적 (파일 경로)
    # ----------------------------
    def track_video(self, video_path: str, visualize: bool = False) -> List[List[Dict[str, Any]]]:
        """
        video_path: 비디오 파일 경로 (ex: "video.mp4")
        visualize: True일 경우 결과를 화면에 그려줌

        변경점: cv2.VideoCapture 반복 대신, 병렬 디코딩 이터레이터로 프레임 공급.
        추론/추적 로직은 기존과 동일하게 '프레임별 순차' 처리.
        """
        results: List[List[Dict[str, Any]]] = []
        self.img_size = None  # 비디오 단위로 초기화 (기존 tracking과 동일 시맨틱)

        # 병렬 디코딩 이터레이터 (프레임 글로벌 순서 보장)
        frame_stream = _iter_frames_parallel(
            video_path,
            cpu_workers=self.cpu_workers,
            chunk_sec=self.chunk_sec
        )

        trails = defaultdict(lambda: deque(maxlen=30)) if visualize else None
        trail_thickness = 2

        for _, frame in frame_stream:
            if frame is None:
                results.append([])
                continue

            # 1프레임 추적 (변수명 동일)
            frame_res = self.track_image(frame)
            results.append(frame_res)

            if visualize:
                vis = self.visualize_frame(frame, frame_res, trails=trails,
                                           trail_len=30, trail_thickness=trail_thickness, copy=True)
                cv2.imshow("tracking", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if visualize:
            cv2.destroyAllWindows()
        return results

    # ----------------------------
    # 기존 호환: tracking() → track_video() 래퍼
    # ----------------------------
    def tracking(self, video_path, visualize=False):
        """
        (호환용) 기존 시그니처 유지.
        내부적으로 track_video() 호출하여 동일 결과 반환.
        """
        return self.track_video(video_path=video_path, visualize=visualize)

    # ----------------------------
    # 기존 호환: trackingVideo() (시각화된 비디오 저장)
    # ----------------------------
    def trackingVideo(self, video_path, save_path, trail_len: int = 30):
        """
        video_path: 입력 영상 경로
        save_path : 저장할 결과 영상 경로
        trail_len: 궤적 길이(저장할 최근 중심점 개수)

        동작: track_video()로 결과 계산 후, 원본 비디오를 순차 읽어 시각화 프레임을 만들어 저장.
        (기존 메서드와 결과 동일)
        """
        # 1) detection+tracking (화면표시 X)
        results = self.track_video(video_path=video_path, visualize=False)

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

        # 3) ID별 궤적 저장소 (변수명 유지)
        trails = defaultdict(lambda: deque(maxlen=trail_len))
        trail_thickness = 2

        # 4) 프레임 순회하며 시각화 + 저장 (변수명/로직 동일)
        res_iter = iter(results)
        for frame_id in range(len(results)):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            try:
                frame_res = next(res_iter)
            except StopIteration:
                break

            vis = self.visualize_frame(frame, frame_res, trails=trails,
                                       trail_len=trail_len, trail_thickness=trail_thickness, copy=True)
            writer.write(vis)

        cap.release()
        writer.release()
        print(f"결과 영상 저장 완료: {save_path}")
        return results
