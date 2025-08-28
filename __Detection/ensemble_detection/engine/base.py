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

        # 인스턴스 생성 (클래스 __init__ 시그니처에 존재할 때만 device 전달)
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

    def _detect_all(self, image) -> List[Any]:
        if self.pool is None:
            return [self._infer_on_right_device(d, image) for d in self.detectors]
        futs = [self.pool.submit(self._infer_on_right_device, d, image) for d in self.detectors]
        return [f.result() for f in futs]

    # mmdet 결과 → 공통 스키마(dict)로 변환
    def _parse(self, result, det) -> List[Dict[str, Any]]:
        parsed: List[Dict[str, Any]] = []
        preds = result.pred_instances
        id2coco = getattr(det, "id2coco", {})
        coco2name = getattr(det, "coco2name", {})

        for i in range(len(preds.labels)):
            score = float(preds.scores[i].item())
            if score < self.thres:
                continue
            lid = int(preds.labels[i].item())
            coco_id = id2coco.get(lid, lid)
            label = coco2name.get(coco_id, f"label_{coco_id}")
            bbox = preds.bboxes[i].tolist()
            parsed.append({
                "class_id": coco_id,
                "label": label,
                "score": score,
                "bbox": bbox,
                "source": getattr(det, "DETECTOR_NAME", "unknown"),
            })
        return parsed

    def detect(self, image) -> List[Dict[str, Any]]:
        results = self._detect_all(image)
        merged: List[Dict[str, Any]] = []
        for det, res in zip(self.detectors, results):
            merged.extend(self._parse(res, det))
        return merged
