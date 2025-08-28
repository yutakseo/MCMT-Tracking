# __Detection/ensemble_detection/worker.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from mmdet.apis import init_detector, inference_detector
from .engine.detector_base import DetectorBase
from .engine.registry import register_detector


@register_detector("worker")
class WorkerDetector(DetectorBase):
    # ---- 기본값(프로젝트 경로/라벨) ----
    DEFAULT_DEVICE = "cuda:1"
    DEFAULT_CONFIG = "/workspace/PretrainedModel_by_JeonYT/worker/yolov8x_signalman.py"
    DEFAULT_CKPT   = "/workspace/PretrainedModel_by_JeonYT/worker/epoch_100.pth"
    DEFAULT_CLASSES: List[str] = ["signalman", "worker"]
    DEFAULT_ID2COCO: Dict[int, int] = {0: 2, 1: 3}

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        config: str = DEFAULT_CONFIG,
        checkpoint: str = DEFAULT_CKPT,
        class_names: Optional[List[str]] = None,
        id2coco_map: Optional[Dict[int, int]] = None,
    ):
        # 1) 클래스/매핑 정의 (인자로 오버라이드 가능)
        self._class_names: List[str] = class_names or self.DEFAULT_CLASSES
        self._id2coco: Dict[int, int] = id2coco_map or self.DEFAULT_ID2COCO

        # 간단 검증: id2coco 키가 클래스 인덱스 범위 내인지 확인
        max_idx = len(self._class_names) - 1
        for k in self._id2coco.keys():
            if not (0 <= int(k) <= max_idx):
                raise ValueError(f"id2coco key {k} is out of range (0..{max_idx}) for class_names.")

        self._coco2name: Dict[int, str] = {v: self._class_names[k] for k, v in self._id2coco.items()}

        # 2) 모델 로드
        self._model = init_detector(config=config, checkpoint=checkpoint, device=device)

    # === 필수 프로퍼티 ===
    @property
    def model(self) -> Any:
        return self._model

    @property
    def id2coco(self) -> Dict[int, int]:
        return self._id2coco

    @property
    def coco2name(self) -> Dict[int, str]:
        return self._coco2name

    # === 필수 메서드 ===
    def detect(self, image: Any) -> Any:
        # mmdet 3.x: result.pred_instances.{labels,scores,bboxes}
        return inference_detector(self._model, image)
