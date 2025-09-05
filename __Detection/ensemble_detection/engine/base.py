# __Detection/ensemble_detection/engine/base.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import torch

from .registry import autodiscover, DETECTOR_REGISTRY, build_detector

# 현재 패키지: '__Detection.ensemble_detection.engine'
# 상위(탐색 대상): '__Detection.ensemble_detection'
_PARENT_PKG = __package__.rsplit('.', 1)[0]
autodiscover(_PARENT_PKG)


def _auto_device_map(detector_names: List[str]) -> Dict[str, str]:
    """
    device_map 미지정 시, 등록된 detector들을 GPU에 라운드로빈 배치.
    GPU가 없으면 모두 'cpu'로 매핑.
    """
    n = torch.cuda.device_count()
    if n > 0:
        return {name: f"cuda:{i % n}" for i, name in enumerate(detector_names)}
    else:
        return {name: "cpu" for name in detector_names}


class EnsembleDetector:
    def __init__(
        self,
        thres: float = 0.3,
        names: Optional[List[str]] = None,           # None이면 등록된 전부
        exclude: Optional[List[str]] = None,         # 제외 목록
        device_map: Optional[Dict[str, str]] = None, # {"vehicle":"cuda:0", ...}
        use_async: bool = True,
        max_workers: Optional[int] = None,
    ):
        self.thres = float(thres)
        self.use_async = use_async

        # 사용할 디텍터 선택
        all_names = sorted(DETECTOR_REGISTRY.keys())
        selected = all_names if names is None else list(names)
        if exclude:
            ex = set(exclude)
            selected = [n for n in selected if n not in ex]
        assert selected, f"No detectors selected. Available: {all_names}"

        # device 매핑 (미지정 시 자동)
        if device_map is None:
            device_map = _auto_device_map(selected)

        # 인스턴스 생성
        self.detectors = []
        for n in selected:
            dev = device_map.get(n)
            det = build_detector(n, device=dev) if dev is not None else build_detector(n)
            self.detectors.append(det)

        # 스레드풀 (2개 이상일 때만)
        self.pool = (
            ThreadPoolExecutor(max_workers=max_workers or len(self.detectors))
            if self.use_async and len(self.detectors) > 1
            else None
        )

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True)
            self.pool = None

    @staticmethod
    def _infer_on_right_device(det, image):
        """스레드별 CUDA 컨텍스트를 해당 모델 디바이스로 고정 후 detect 호출."""
        try:
            dev = next(det.model.parameters()).device
        except Exception:
            dev = getattr(det.model, "device", torch.device("cpu"))
        if isinstance(dev, torch.device) and dev.type == "cuda":
            torch.cuda.set_device(dev)
        return det.detect(image)

    @staticmethod
    def _infer_batch_on_right_device(det, images: List[Any]):
        """스레드별 CUDA 컨텍스트 고정 후 detect_batch 또는 프레임별 detect 호출."""
        try:
            dev = next(det.model.parameters()).device
        except Exception:
            dev = getattr(det.model, "device", torch.device("cpu"))
        if isinstance(dev, torch.device) and dev.type == "cuda":
            torch.cuda.set_device(dev)

        if hasattr(det, "detect_batch"):
            return det.detect_batch(images)  # 기대: List[wrapper] or List[List[dict]] or List[Tensor]
        else:
            # 폴백: 프레임별 detect
            return [det.detect(img) for img in images]

    def _detect_all(self, image) -> List[Any]:
        """단일 프레임: 모든 디텍터 호출(병렬 가능)"""
        if self.pool is None:
            return [self._infer_on_right_device(d, image) for d in self.detectors]
        futs = [self.pool.submit(self._infer_on_right_device, d, image) for d in self.detectors]
        return [f.result() for f in futs]

    def _detect_all_batch(self, images: List[Any]) -> List[List[Any]]:
        """
        배치 프레임: 모든 디텍터 호출(병렬 가능)
        Returns:
            List (per-detector) of List (per-frame) of result
        """
        if self.pool is None:
            return [self._infer_batch_on_right_device(d, images) for d in self.detectors]
        futs = [self.pool.submit(self._infer_batch_on_right_device, d, images) for d in self.detectors]
        return [f.result() for f in futs]

    # ---------- 파서들 ----------
    def _parse_wrapper(self, result, det) -> List[Dict[str, Any]]:
        """
        mmdet / custom wrapper → 공통 스키마(dict)로 변환
        result: wrapper with .pred_instances.{bboxes,scores,labels}
        """
        parsed: List[Dict[str, Any]] = []
        preds = result.pred_instances
        id2coco = getattr(det, "id2coco", {})
        coco2name = getattr(det, "coco2name", {})

        # 일부 구현체는 텐서/ndarray 혼용 → 일괄 cpu().numpy() 사용 가능
        labels = preds.labels
        scores = preds.scores
        bboxes = preds.bboxes

        # 길이 0인 경우
        if len(labels) == 0:
            return parsed

        for i in range(len(labels)):
            score = float(scores[i].item())
            if score < self.thres:
                continue
            lid = int(labels[i].item())
            coco_id = id2coco.get(lid, lid)
            label = coco2name.get(coco_id, f"label_{coco_id}")
            bbox = bboxes[i].tolist()
            parsed.append({
                "class_id": coco_id,
                "label": label,
                "score": score,
                "bbox": bbox,
                "source": getattr(det, "DETECTOR_NAME", "unknown"),
            })
        return parsed

    def _parse_tensor(self, t: torch.Tensor, det) -> List[Dict[str, Any]]:
        """
        (N,6) 텐서 → 공통 스키마(dict)로 변환
        columns: x1,y1,x2,y2,score,class_id  (class_id는 coco id라고 가정)
        """
        parsed: List[Dict[str, Any]] = []
        if t is None or t.numel() == 0:
            return parsed

        # 안전 변환 (CPU/float32)
        if t.is_cuda:
            t = t.detach().cpu()
        else:
            t = t.detach()
        if t.dtype != torch.float32:
            t = t.to(torch.float32)

        coco2name = getattr(det, "coco2name", {})
        for row in t.tolist():
            x1, y1, x2, y2, score, cid = row
            score = float(score)
            if score < self.thres:
                continue
            cid = int(cid)
            label = coco2name.get(cid, f"label_{cid}")
            parsed.append({
                "class_id": cid,
                "label": label,
                "score": score,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "source": getattr(det, "DETECTOR_NAME", "unknown"),
            })
        return parsed

    def _parse_any(self, result: Any, det) -> List[Dict[str, Any]]:
        """
        result 타입에 따라 파싱:
        - wrapper(pred_instances 보유)
        - list[dict] (이미 표준 스키마) → 스코어 필터만 적용
        - Tensor(N,6)
        """
        # wrapper 케이스
        if hasattr(result, "pred_instances"):
            return self._parse_wrapper(result, det)

        # 이미 dict 스키마인 경우
        if isinstance(result, list) and (len(result) == 0 or isinstance(result[0], dict)):
            if not result:
                return []
            # score threshold만 재확인
            th = self.thres
            out = []
            src = getattr(det, "DETECTOR_NAME", "unknown")
            for r in result:
                try:
                    if float(r.get("score", 0.0)) >= th:
                        rr = dict(r)
                        rr.setdefault("source", src)
                        out.append(rr)
                except Exception:
                    continue
            return out

        # 텐서 케이스
        if isinstance(result, torch.Tensor):
            return self._parse_tensor(result, det)

        # 지원 외 타입 → 빈 결과
        return []

    # ---------- 공개 API ----------
    def detect(self, image) -> List[Dict[str, Any]]:
        """단일 프레임 추론 → 병합 결과(list[dict])"""
        results = self._detect_all(image)
        merged: List[Dict[str, Any]] = []
        for det, res in zip(self.detectors, results):
            merged.extend(self._parse_any(res, det))
        # 메모리 참조 줄이기
        del results
        return merged

    def detect_batch(self, images: List[Any]) -> List[List[Dict[str, Any]]]:
        """
        배치 프레임 추론 → 프레임별 병합 결과
        Returns:
            List (per-frame) of List[dict]
        """
        if not images:
            return []

        # per-detector results: List[ per-detector -> per-frame -> result ]
        per_det = self._detect_all_batch(images)

        # 프레임 수
        T = len(images)
        merged_per_frame: List[List[Dict[str, Any]]] = [[] for _ in range(T)]

        # 디텍터마다, 프레임마다 파싱/머지
        for det, det_batch in zip(self.detectors, per_det):
            # det_batch: 길이 T
            if not isinstance(det_batch, list) or len(det_batch) != T:
                # 불일치시 방어적으로 프레임별 detect로 폴백
                for i, img in enumerate(images):
                    res = self._infer_on_right_device(det, img)
                    merged_per_frame[i].extend(self._parse_any(res, det))
                continue

            for i, res in enumerate(det_batch):
                merged_per_frame[i].extend(self._parse_any(res, det))

        # 메모리 참조 줄이기
        del per_det
        return merged_per_frame
