#/workspace/__Detection/ensemble_detection/ultra_detector.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from ultralytics import YOLO
from types import SimpleNamespace
import torch

from .engine.detector_base import DetectorBase
from .engine.registry import register_detector


@register_detector("ultra")
class VehicleDetector(DetectorBase):
    DETECTOR_NAME = "ultra"  # source 필드용
    DEFAULT_DEVICE = "cuda"
    DEFAULT_CKPT   = "/workspace/PretrainedModel_by_JeonYT/new_weights/250812_yolo11x_whhs_con8_5010_5016_26cls_test_비교용/con8_20250812_100020/weights/best.pt"

    # 전체 26 클래스
    DEFAULT_CLASSES: List[str] = [
        "worker", "worker_without_vest", "signalman", "signalman_with_baton_non_red",
        "helmet", "head", "small_worker", "small_worker_without_vest",
        "small_signalman", "small_signalman_with_baton_non_red", "small_helmet",
        "small_head", "non_worker", "non_helmet", "non_head",
        "excavator", "dump_truck", "crane_mobile", "tower_crane", "cargo_truck",
        "forklift", "mixer_truck", "dozer", "scissor_lift",
        "non_excavator", "non_dump_truck"
    ]

    # 관심 클래스만 COCO ID로 매핑 (예시)
    DEFAULT_ID2COCO: Dict[int, int] = {
        1 : 1,    # worker
        2 : 1,
        3 : 1,
        7 : 1,
        15: 10,   # excavator
        16: 11,   # dump_truck
        20: 12,   # forklift
        21: 13,   # mixer_truck
        23: 14,   # scissor_lift
        22: 16,   # dozer
        19: 17,   # cargo_truck
        17: 18,   # crane_mobile
    }

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        checkpoint: str = DEFAULT_CKPT,
        class_names: Optional[List[str]] = None,
        id2coco_map: Optional[Dict[int, int]] = None,
    ):
        # 클래스/매핑 정의
        self._class_names: List[str] = class_names or self.DEFAULT_CLASSES
        self._id2coco: Dict[int, int] = id2coco_map or self.DEFAULT_ID2COCO
        self._coco2name: Dict[int, str] = {
            v: self._class_names[k] for k, v in self._id2coco.items()
        }

        # 모델 로드
        self._model = YOLO(checkpoint)
        self._device = device
        print(f"[DEBUG] VehicleDetector loaded with Ultralytics YOLO on {device}")

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

    # === detect() ===
    def detect(self, image: Any):
        """
        Ultralytics YOLO 추론 → MMDetection 스타일 wrapper 반환
        (pred_instances.bboxes, scores, labels)
        """
        results = self._model.predict(source=image, device=self._device, verbose=False)
        r = results[0]
        boxes = r.boxes

        # EnsembleDetector._parse() 가 요구하는 구조 흉내내기
        pred_instances = SimpleNamespace(
            bboxes=torch.as_tensor(boxes.xyxy.cpu().numpy(), dtype=torch.float32),
            scores=torch.as_tensor(boxes.conf.cpu().numpy(), dtype=torch.float32),
            labels=torch.as_tensor(boxes.cls.cpu().numpy(), dtype=torch.int64),
        )

        wrapper = SimpleNamespace(pred_instances=pred_instances)
        return wrapper
