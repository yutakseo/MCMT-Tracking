# ensemble_detection/registry.py
from typing import Dict, Type, Callable
import importlib, pkgutil, inspect

DETECTOR_REGISTRY: Dict[str, Type] = {}

def register_detector(name: str) -> Callable[[Type], Type]:
    """클래스 위에 붙여 자동 등록: @register_detector("vehicle")"""
    def deco(cls: Type) -> Type:
        DETECTOR_REGISTRY[name] = cls
        setattr(cls, "DETECTOR_NAME", name)
        return cls
    return deco


def autodiscover(package_name: str) -> None:
    """
    패키지 내 모듈을 전부 import하여 @register_detector 데코레이터가 실행되게 함.
    예: autodiscover(" __Detection.ensemble_detection ")
    """
    pkg = importlib.import_module(package_name)
    for _, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
        if not ispkg:
            importlib.import_module(f"{package_name}.{modname}")


def build_detector(name: str, **kwargs):
    """시그니처를 보고 device 같은 인자만 골라 전달하여 안전 생성"""
    cls = DETECTOR_REGISTRY[name]
    sig = inspect.signature(cls.__init__)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**allowed)
