# _base_.py
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple
import torch

from .vehicle import VehicleDetector
from .worker import WorkerDetector


def _get_model_device(detector) -> torch.device:
    """detector.model 에서 안전하게 디바이스 얻기"""
    try:
        # PyTorch 표준
        return next(detector.model.parameters()).device
    except Exception:
        # mmdet/mmengine에선 model.device가 있을 수도 있음
        dev = getattr(detector.model, "device", None)
        if isinstance(dev, torch.device):
            return dev
        return torch.device("cpu")


def _infer_on_its_device(detector, image):
    """해당 detector가 올라간 디바이스 컨텍스트에서 detect 호출"""
    dev = _get_model_device(detector)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    return detector.detect(image)


class EnsembleDetector:
    def __init__(self, thres: float = 0.3, use_async: bool = True, max_workers: int = 2):
        self.vehicle = VehicleDetector()  # 내부에서 device="cuda:0" 고정해둔 상태
        self.worker  = WorkerDetector()   # 내부에서 device="cuda:1" 고정해둔 상태
        self.thres   = thres
        self.use_async = use_async
        self._pool = ThreadPoolExecutor(max_workers=max_workers) if use_async else None

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _detect(self, image) -> Tuple[Any, Any]:
        if not self.use_async or self._pool is None:
            v = _infer_on_its_device(self.vehicle, image)
            w = _infer_on_its_device(self.worker,  image)
            return v, w

        f_vehicle = self._pool.submit(_infer_on_its_device, self.vehicle, image)
        f_worker  = self._pool.submit(_infer_on_its_device, self.worker,  image)
        return f_vehicle.result(), f_worker.result()

    def _parse_detsample(self, sub_result, id2coco=None, coco2name=None) -> List[Dict[str, Any]]:
        parsed = []
        preds = sub_result.pred_instances
        for i in range(len(preds.labels)):
            score = preds.scores[i].item()
            if score < self.thres:
                continue
            label_id = preds.labels[i].item()
            coco_id = id2coco.get(label_id, label_id) if id2coco else label_id
            label = coco2name.get(coco_id, f"unknown_{coco_id}") if coco2name else f"label_{coco_id}"
            bbox = preds.bboxes[i].tolist()
            parsed.append({
                "class_id": coco_id,
                "label": label,
                "score": score,
                "bbox": bbox
            })
        return parsed

    def detect(self, image) -> List[Dict[str, Any]]:
        det_v, det_w = self._detect(image)

        vehicle = self._parse_detsample(
            det_v, id2coco=self.vehicle.id2coco, coco2name=self.vehicle.coco2name
        )
        worker = self._parse_detsample(
            det_w, id2coco=self.worker.id2coco, coco2name=self.worker.coco2name
        )
        return vehicle + worker
