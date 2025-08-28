# __Detection/ensemble_detection/engine/detector_base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
import torch

class DetectorBase(ABC):
    """모든 Detector는 이 인터페이스를 상속해야 함."""
    @abstractmethod
    def detect(self, image: Any) -> Any: ...
    @property
    @abstractmethod
    def id2coco(self) -> Dict[int, int]: ...
    @property
    @abstractmethod
    def coco2name(self) -> Dict[int, str]: ...
    @property
    @abstractmethod
    def model(self) -> Any: ...
    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except Exception:
            return getattr(self.model, "device", torch.device("cpu"))
