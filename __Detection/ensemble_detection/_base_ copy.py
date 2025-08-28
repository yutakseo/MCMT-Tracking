# ensemble_detection/_base_.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from _registry_ import autodiscover, DETECTOR_REGISTRY, build_detector
import torch

# 패키지 내 모듈 자동 import → 데코레이터 실행 → 레지스트리 채워짐
# "__Detection.ensemble_detection._base_" → 앞부분 패키지명만 추출
_PACKAGE = __name__.rsplit('.', 1)[0]
autodiscover(_PACKAGE)

class EnsembleDetector:
    def __init__(
        self,
        names: List[str],                 # ["vehicle","worker", ...]
        device_map: Optional[Dict[str, str]] = None,  # {"vehicle":"cuda:0","worker":"cuda:1"}
        thres: float = 0.3,
        use_async: bool = True,
        max_workers: Optional[int] = None,
    ):
        self.thres = float(thres)
        self.use_async = use_async

        # 존재 확인
        for n in names:
            assert n in DETECTOR_REGISTRY, f"Detector '{n}' not registered."

        # 인스턴스 생성 (클래스 __init__ 시그니처에 맞춰 device 전달)
        self.detectors = []
        for n in names:
            dev = device_map.get(n) if device_map else None
            det = build_detector(n, device=dev) if dev else build_detector(n)
            self.detectors.append(det)

        self.pool = ThreadPoolExecutor(
            max_workers=max_workers or len(self.detectors)
        ) if use_async and len(self.detectors) > 1 else None

    def close(self):
        if self.pool:
            self.pool.shutdown(wait=True)
            self.pool = None

    @staticmethod
    def _infer_on_right_device(det, image):
        # 안전하게 스레드별 CUDA 컨텍스트 고정
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

    # mmdet 결과 → 공통 스키마
    def _parse(self, result, det) -> List[Dict[str, Any]]:
        parsed = []
        preds = result.pred_instances
        for i in range(len(preds.labels)):
            score = float(preds.scores[i].item())
            if score < self.thres:
                continue
            lid = int(preds.labels[i].item())
            coco_id = det.id2coco.get(lid, lid)
            label = det.coco2name.get(coco_id, f"label_{coco_id}")
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
