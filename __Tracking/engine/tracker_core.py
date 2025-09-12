# /workspace/__Tracking/core/tracker_core.py
from __Tracking.ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class TrackerCore:
    def __init__(self, args, detector) -> None:
        self.args = args
        self.detector = detector
        self.tracker: Optional[BYTETracker] = None
        self.img_size: Optional[Tuple[int, int]] = None

        # detector에서 class_id → name 맵 추출 (있다면)
        if hasattr(detector, "name_map"):
            self.class_map = detector.name_map()
        else:
            self.class_map = {}

        # 초기화
        self.reset_tracker()

    def reset_tracker(self):
        """트래커 상태를 초기화 (새 영상 추적 시작 시 호출)"""
        self.tracker = BYTETracker(self.args)
        self.img_size = None

        # detector가 동적으로 교체되거나 맵이 바뀔 수 있으니 한 번 더 동기화
        if hasattr(self.detector, "name_map"):
            try:
                self.class_map = self.detector.name_map()
            except Exception:
                # 실패해도 추적은 계속
                pass

    # ----------------------------
    # 기존: 단일 프레임 추적
    # ----------------------------
    def track_frame(self, frame) -> List[Dict[str, Any]]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        fh, fw = frame.shape[:2]
        
        if self.img_size is None:
            # 첫 프레임 기준으로 img_size 고정
            self.img_size = (fh, fw)
        elif self.img_size != (fh, fw):
            # 예상치 못한 해상도 변경: 트래커를 재초기화해서 스케일 꼬임 방지
            self.reset_tracker()
            self.img_size = (fh, fw)

        # 1) DetectionAPI → torch.Tensor (N,6)
        dets = self.detector.detect(frame)

        # 2) ByteTrack 업데이트
        online_targets = self.tracker.update(dets, (fh, fw), self.img_size)

        # 3) 결과 변환
        results = self._to_results(online_targets)
        return results

    # ----------------------------
    # 신규: 배치 프레임 추적
    # ----------------------------
    def track_video_batch(self, frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        frames: List of np.ndarray(BGR)
        return: List of per-frame results
        """
        if not frames:
            return []

        # 유효 프레임만 필터(간혹 None/빈 프레임 섞일 수 있음)
        valid_frames: List[np.ndarray] = []
        for f in frames:
            if isinstance(f, np.ndarray) and f.size > 0:
                valid_frames.append(f)
            else:
                valid_frames.append(None)  # 자리 유지 (출력 정렬 목적)

        # 첫 유효 프레임 해상도 확정
        base_hw: Optional[Tuple[int, int]] = None
        for f in valid_frames:
            if f is not None:
                base_hw = f.shape[:2]
                break
        if base_hw is None:
            # 모두 무효 프레임
            return [[] for _ in frames]

        fh, fw = base_hw
        if self.img_size is None:
            self.img_size = (fh, fw)
        elif self.img_size != (fh, fw):
            # 배치 내 첫 유효 프레임이 기존과 해상도 다르면 트래커 재초기화
            self.reset_tracker()
            self.img_size = (fh, fw)

        # 1) 배치 검출 지원 시 활용, 아니면 per-frame 폴백
        dets_list: List[Any] = []
        if hasattr(self.detector, "detect_batch"):
            try:
                # detect_batch는 유효 프레임만 전달
                compact_frames = [f for f in valid_frames if f is not None]
                compact_dets = self.detector.detect_batch(compact_frames)

                # compact_dets를 원래 인덱스로 재매핑
                it = iter(compact_dets)
                for f in valid_frames:
                    if f is None:
                        dets_list.append(None)
                    else:
                        dets_list.append(next(it))
            except Exception:
                # 배치 실패 시 안전 폴백
                dets_list = [self.detector.detect(f) if f is not None else None for f in valid_frames]
        else:
            dets_list = [self.detector.detect(f) if f is not None else None for f in valid_frames]

        # 2) 순차 업데이트 (트래커는 시계열 의존)
        batch_results: List[List[Dict[str, Any]]] = []
        for f, dets in zip(valid_frames, dets_list):
            if f is None or dets is None:
                batch_results.append([])
                continue

            h, w = f.shape[:2]
            if (h, w) != self.img_size:
                # 배치 중간에라도 해상도 바뀌면 재초기화 (안전)
                self.reset_tracker()
                self.img_size = (h, w)

            online_targets = self.tracker.update(dets, (h, w), self.img_size)
            batch_results.append(self._to_results(online_targets))

        return batch_results

    # ----------------------------
    # 내부 변환 함수
    # ----------------------------
    def _to_results(self, online_targets) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for t in online_targets:
            class_id = t.class_id if getattr(t, "class_id", None) is not None else -1
            label = self.class_map.get(class_id, class_id)
            
            # tensor를 numpy로 변환
            bbox = t.tlwh
            if hasattr(bbox, 'cpu'):  # torch tensor인 경우
                bbox = bbox.cpu().numpy()
            elif not isinstance(bbox, np.ndarray):  # 다른 타입인 경우
                bbox = np.array(bbox, dtype=np.float32)
            
            score = t.score
            if hasattr(score, 'cpu'):  # torch tensor인 경우
                score = score.cpu().item()
            elif not isinstance(score, (int, float)):  # numpy scalar인 경우
                score = float(score)
            
            results.append({
                "id": int(t.track_id),          # int로 변환
                "class_id": int(class_id),      # int로 변환
                "label": str(label),            # str로 변환
                "bbox": bbox,                   # numpy array
                "score": float(score),          # float로 변환
            })
        return results
