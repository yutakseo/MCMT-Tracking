# MCMT Engine - Multi-Camera Multi-Target Tracking Engine
"""
멀티카메라 다중 객체 추적 엔진

주요 모듈:
- core: 핵심 추론 엔진 및 매핑
- streaming: 스트리밍 및 카메라 처리
- visualization: 시각화 및 렌더링
- monitoring: GPU 및 시스템 모니터링
"""

from .core.async_inference import AsyncEngine
from .SCST.stream_SCST import streamSCST, SCST
from .SCST.cam_stream import CamMJPEG
from .visualization.visualizer import PlanVisualizer
from .monitoring.gpu_monitor import GPUMonitor

__all__ = [
    'AsyncEngine',
    'streamSCST', 'SCST', 
    'CamMJPEG',
    'PlanVisualizer',
    'GPUMonitor'
]

class MCMT:
    def __init__(self):
        streamimg:bool
        
