# __Detection/ensemble_detection/engine/registry.py
from __future__ import annotations
from typing import Dict, Type, Callable, Any
import importlib, pkgutil, inspect
from .detector_base import DetectorBase

DETECTOR_REGISTRY: Dict[str, Type[DetectorBase]] = {}

def register_detector(name: str) -> Callable[[Type[DetectorBase]], Type[DetectorBase]]:
    if not name or not isinstance(name, str):
        raise ValueError("Detector name must be a non-empty string.")
    def deco(cls: Type[DetectorBase]) -> Type[DetectorBase]:
        if not issubclass(cls, DetectorBase):
            raise TypeError(f"{cls.__name__} must subclass DetectorBase to be registered as '{name}'.")
        if name in DETECTOR_REGISTRY and DETECTOR_REGISTRY[name] is not cls:
            raise ValueError(f"Detector name '{name}' already registered with {DETECTOR_REGISTRY[name]}; got {cls}.")
        DETECTOR_REGISTRY[name] = cls
        setattr(cls, "DETECTOR_NAME", name)
        return cls
    return deco

def autodiscover(package_name: str) -> None:
    """ensemble_detection 패키지(=engine의 상위) 안의 모듈들만 import해서 데코레이터 실행."""
    pkg = importlib.import_module(package_name)
    for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
        # engine 서브패키지는 건너뜀, 파일(.py)만 import
        if ispkg:
            continue
        importlib.import_module(f"{package_name}.{modname}")

def build_detector(name: str, **kwargs) -> DetectorBase:
    if name not in DETECTOR_REGISTRY:
        available = ", ".join(sorted(DETECTOR_REGISTRY.keys()))
        raise KeyError(f"Detector '{name}' not registered. Available: [{available}]")
    cls = DETECTOR_REGISTRY[name]
    sig = inspect.signature(cls.__init__)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    inst = cls(**allowed)
    # 런타임 계약 검사
    for attr in ("id2coco", "coco2name", "model", "detect"):
        if not hasattr(inst, attr):
            raise AttributeError(f"{cls.__name__} missing required attr/method: '{attr}'")
    return inst

def list_detectors() -> list[str]:
    return sorted(DETECTOR_REGISTRY.keys())
