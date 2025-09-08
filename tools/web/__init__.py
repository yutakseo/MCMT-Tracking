# /workspace/tools/web/__init__.py
"""
웹 관련 모듈들
- server: FastAPI 웹 서버
- utils: 웹 관련 유틸리티 함수들
- viz: 웹 시각화 관련 기능
- webviz: 웹 플랜 시각화
- app: 웹 앱 러너 (엔진 연동)
- manager: 웹 서버 관리자
"""

from .server import app, set_webviz, set_cam_streams, set_cam_overlays, set_class_map
from .utils import WebOverlayManager, WebStreamManager
from .planviz import PlanViz
from .streamviz import StreamViz
from .app import MultiCameraWebApp
from .manager import WebServerManager

__all__ = [
    "app",
    "set_webviz",
    "set_cam_streams",
    "set_cam_overlays",
    "set_class_map",
    "WebOverlayManager",
    "WebStreamManager",
    "PlanViz",
    "StreamViz",
    "MultiCameraWebApp",
    "WebServerManager",
]
