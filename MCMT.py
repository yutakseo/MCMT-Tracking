# /workspace/MCMT.py - Multi-Camera Multi-Target Tracking Core
"""
멀티카메라 다중 객체 추적 핵심 시스템
- 카메라 초기화 및 호모그래피 설정
- 공유 모델 관리
- 추론 엔진 초기화
- 웹 시각화 연동
"""

import os
import logging
import time
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

sys.path.append("/workspace")

from mmyolo.utils import register_all_modules
register_all_modules()

from MCMT_engine.core.async_inference import AsyncEngine
from MCMT_engine.streaming.stream_SCST import streamSCST
from MCMT_engine.streaming.cam_stream import CamMJPEG
from MCMT_engine.visualization.visualizer import PlanVisualizer
from tools.webviz import WebPlanViz
from __Detection.detection_api import DetectionAPI
from __Tracking.tracking_api import TrackerAPI

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

# FFmpeg/RTSP 옵션 (지연 감소, 타임아웃 설정)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|buffer_size;102400|max_delay;0|stimeout;5000000"
)

# =============================================================================
# 설정 상수
# =============================================================================
PLAN_PATH = "/workspace/assets/250904_homograph_coordinate-plane2.jpg"

# RTSP 스트림 주소 (실시간 CCTV)
CAMERA_SOURCES = [
    "rtsp://210.99.70.120:1935/live/cctv001.stream",  # Camera 1
    "rtsp://210.99.70.120:1935/live/cctv001.stream",  # Camera 2
    "rtsp://210.99.70.120:1935/live/cctv001.stream",  # Camera 3
]

# 호모그래피 설정
PLAN_POINTS = [
    (1170, 214), (1170, 559), (1170, 904),
    (2212, 214), (2212, 559), (2212, 904),
    (3255, 214), (3255, 559), (3255, 904)
]

# 각 카메라별 호모그래피 포인트
CAMERA_POINTS = [
    [(1033,475), (948,474), (863,473), (1019,527), (890,524), (769,519), (973,667), (741,652), (548,620)],  # Camera 1
    [(518,466), (430,471), (341,474), (613,510), (485,519), (354,527), (829,608), (634,645), (397,668)],   # Camera 2
    [(357,602), (566,648), (832,683), (620,498), (754,509), (893,513), (726,453), (819,456), (911,459)]    # Camera 3
]

class TrackingArgs:
    """추적 시스템 설정"""
    track_thresh = 0.3
    match_thresh = 0.9
    track_buffer = 180
    mot20 = False
    cpu_workers = 20
    chunk_sec = 5.0
    batch_size = 10

# =============================================================================
# 핵심 멀티카메라 추적 시스템
# =============================================================================
class MultiCameraTrackingSystem:
    """멀티카메라 다중 객체 추적 핵심 시스템"""
    
    def __init__(self, detector_models: List[str] = None):
        self.shared_detector = None
        self.shared_tracker = None
        self.cameras = []
        self.engine = None
        self.viz = None
        self.streams = None
        self.detector_models = detector_models or ["vehicle", "ultra_people"]
        
    def initialize_shared_models(self) -> bool:
        """단일 모델 인스턴스 생성 (모든 카메라가 공유)"""
        try:
            print("📦 공유 모델 초기화 중...")
            args = TrackingArgs()
            print(f"🎯 사용할 탐지 모델: {self.detector_models}")

            self.shared_detector = DetectionAPI(
                models=self.detector_models,
                thres=0.1,  # 임계값을 낮춤
                device="cuda:0",
                use_async=True,
                max_workers=1
            )

            self.shared_tracker = TrackerAPI(args=args, detector=self.shared_detector)
            print("✅ 공유 모델 초기화 완료")
            
            # GPU 메모리 사용량 확인
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"💾 GPU 메모리 사용량: {allocated:.2f}GB (공유 모델)")
            
            return True
        except Exception as e:
            print(f"❌ 공유 모델 초기화 실패: {e}")
            return False
    
    def initialize_cameras(self) -> bool:
        """카메라 초기화 (단일 모델 공유)"""
        try:
            print("📷 카메라 초기화 중...")
            print(f"🎥 카메라 소스: {CAMERA_SOURCES}")
            
            args = TrackingArgs()
            self.cameras = []
            
            for i, (source, cam_pts) in enumerate(zip(CAMERA_SOURCES, CAMERA_POINTS)):
                try:
                    cam = streamSCST(
                        source,
                        cam_pts,
                        PLAN_PATH,
                        PLAN_POINTS,
                        args,
                        detector=self.shared_detector,  # 공유 모델 주입
                        tracker=self.shared_tracker     # 공유 트래커 주입
                    )
                    self.cameras.append(cam)
                    print(f"✅ Camera {i+1} 초기화 완료 (소스: {source})")
                except Exception as e:
                    print(f"❌ Camera {i+1} 초기화 실패: {e}")
                    self.cameras.append(None)

            # 성공한 카메라만 필터링
            self.cameras = [cam for cam in self.cameras if cam is not None]
            print(f"🎯 총 {len(self.cameras)}개 카메라가 단일 모델을 공유합니다!")
            
            return len(self.cameras) > 0
        except Exception as e:
            print(f"❌ 카메라 초기화 실패: {e}")
            return False
    
    def initialize_streams(self) -> bool:
        """웹 스트리밍용 카메라 스트림 초기화"""
        try:
            print("📹 웹 스트림 초기화 중...")
            self.streams = {}
            
            for i, source in enumerate(CAMERA_SOURCES):
                try:
                    print(f"🔗 Stream {i+1} 연결 시도 중... (소스: {source})")
                    
                    stream = CamMJPEG(
                        name=f"cam{i+1}",
                        url=source,
                        width=None,  # 원본 크기 유지
                        jpeg_quality=85
                    )
                    stream = stream.start()
                    
                    # 연결 테스트 (3초 대기)
                    print(f"⏳ Stream {i+1} 연결 테스트 중...")
                    time.sleep(3)
                    
                    if getattr(stream, "_jpg", None) and len(stream._jpg) > 1000:
                        self.streams[f"cam{i+1}"] = stream
                        print(f"✅ Stream {i+1} 초기화 완료 (소스: {source})")
                    else:
                        print(f"⚠️ Stream {i+1} 연결 불안정 - 재시도 중...")
                        stream.stop()
                        time.sleep(2)
                        stream = CamMJPEG(name=f"cam{i+1}", url=source, width=480, jpeg_quality=85).start()
                        time.sleep(3)
                        
                        if getattr(stream, "_jpg", None) and len(stream._jpg) > 1000:
                            self.streams[f"cam{i+1}"] = stream
                            print(f"✅ Stream {i+1} 재연결 성공")
                        else:
                            print(f"❌ Stream {i+1} 연결 실패 - 건너뜀")
                            
                except Exception as e:
                    print(f"❌ Stream {i+1} 초기화 실패: {e}")
            
            print(f"✅ {len(self.streams)}개 카메라 스트림이 초기화되었습니다")
            return True
        except Exception as e:
            print(f"❌ 스트림 초기화 실패: {e}")
            return False
    
    def initialize_visualization(self) -> bool:
        """웹 시각화 초기화"""
        try:
            print("🎨 웹 시각화 초기화 중...")
            
            # 도면 파일 존재 확인
            if not os.path.exists(PLAN_PATH):
                print(f"❌ 도면 파일이 없습니다: {PLAN_PATH}")
                return False
                
            self.viz = WebPlanViz(plan_path=PLAN_PATH, show_cam_points=False, fps_limit=12.0)
            print("✅ WebPlanViz 초기화 완료")
            
            return True
        except Exception as e:
            print(f"❌ 웹 시각화 초기화 실패: {e}")
            return False
    
    def initialize_engine(self) -> bool:
        """비동기 추론 엔진 초기화"""
        try:
            print("⚙️ 추론 엔진 초기화 중...")
            self.engine = AsyncEngine(
                self.cameras,
                interval=0.3,
                gpu_warning=90.0,      # 90% 이상: 프레임 스킵
                gpu_danger=95.0,       # 95% 이상: 추론 스킵
                gpu_critical=98.0,     # 98% 이상: 완전 정지
                gpu_recovery=85.0,     # 85% 이하: 정상 복구
                gpu_id=0
            )
            print("✅ 추론 엔진 초기화 완료")
            return True
        except Exception as e:
            print(f"❌ 추론 엔진 초기화 실패: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """시스템 정보 반환"""
        return {
            "cameras": len(self.cameras),
            "streams": len(self.streams) if self.streams else 0,
            "has_detector": self.shared_detector is not None,
            "has_tracker": self.shared_tracker is not None,
            "has_engine": self.engine is not None,
            "has_viz": self.viz is not None,
            "detector_models": self.detector_models,
            "camera_sources": CAMERA_SOURCES,
            "plan_path": PLAN_PATH
        }
    
    def cleanup(self):
        """리소스 정리"""
        print("🧹 리소스 정리 중...")
        
        # 카메라 정리
        for cam in self.cameras:
            try:
                cam.close()
            except:
                pass
        
        # 스트림 정리
        if self.streams:
            for stream in self.streams.values():
                try:
                    stream.stop()
                except:
                    pass
        
        # 공유 모델 정리
        try:
            del self.shared_detector, self.shared_tracker
        except:
            pass
        
        # GPU 메모리 정리
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
        
        print("✅ 정리 완료")

# =============================================================================
# 팩토리 함수
# =============================================================================
def create_tracking_system(detector_models: List[str] = None) -> MultiCameraTrackingSystem:
    """멀티카메라 추적 시스템 생성
    
    Args:
        detector_models: 사용할 탐지 모델 리스트 (예: ["vehicle"], ["ultra_people"], ["vehicle", "ultra_people"])
    """
    system = MultiCameraTrackingSystem(detector_models=detector_models)
    
    # 초기화 순서
    if not system.initialize_shared_models():
        raise RuntimeError("공유 모델 초기화 실패")
    
    if not system.initialize_cameras():
        raise RuntimeError("카메라 초기화 실패")
    
    if not system.initialize_streams():
        print("⚠️ 스트림 초기화 실패 - 웹 스트리밍 비활성화")
    
    if not system.initialize_visualization():
        print("⚠️ 웹 시각화 초기화 실패 - 웹 시각화 비활성화")
    
    if not system.initialize_engine():
        raise RuntimeError("추론 엔진 초기화 실패")
    
    return system

# =============================================================================
# 테스트 및 예제
# =============================================================================
async def test_system(detector_models: List[str] = None):
    """시스템 테스트
    
    Args:
        detector_models: 사용할 탐지 모델 리스트
    """
    try:
        system = create_tracking_system(detector_models=detector_models)
        print("🎉 시스템 초기화 완료!")
        print(f"📊 시스템 정보: {system.get_system_info()}")
        
        # 간단한 테스트 실행
        print("🧪 추론 엔진 테스트 중...")
        count = 0
        async for result in system.engine.stream():
            count += 1
            print(f"Round {count}: {result.get('total_detections', 0)} detections, {result.get('total_tracks', 0)} tracks")
            
            if count >= 5:  # 5라운드만 테스트
                break
        
        print("✅ 시스템 테스트 완료!")
        return system
        
    except Exception as e:
        print(f"❌ 시스템 테스트 실패: {e}")
        return None
    finally:
        if 'system' in locals():
            system.cleanup()

if __name__ == "__main__":
    # 사용 예제:
    # 1. 기본 설정 (vehicle + ultra_people)
    # asyncio.run(test_system())
    
    # 2. 차량만 탐지
    # asyncio.run(test_system(detector_models=["vehicle"]))
    
    # 3. 사람만 탐지  
    asyncio.run(test_system(detector_models=["ultra_people", "worker"]))
    
    # 4. 사용자 정의 모델 조합
    # asyncio.run(test_system(detector_models=["vehicle", "ultra_people"]))
    
    # 기본 테스트 실행
    #asyncio.run(test_system())