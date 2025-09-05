# /workspace/main.py - 멀티카메라 단일 모델 공유 시스템 (웹 자동 실행)

import os
import logging
import time
import asyncio
import sys
import threading
import subprocess
import webbrowser
from pathlib import Path
sys.path.append("/workspace")

from mmyolo.utils import register_all_modules
register_all_modules()

from MCMT_engine.stream_SCST import streamSCST
from MCMT_engine.async_inference import AsyncEngine
from tools.webviz import WebPlanViz
from app_web import set_webviz, set_cam_streams
from __Detection.detection_api import DetectionAPI
from __Tracking.tracking_api import TrackerAPI
from MCMT_engine.cam_stream import CamMJPEG

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

# FFmpeg/RTSP 옵션 (지연 감소, 타임아웃 설정)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|buffer_size;102400|max_delay;0|stimeout;5000000"
)

# =============================================================================
# 설정 - 실시간 RTSP 스트림
# =============================================================================
PLAN_PATH = "/workspace/assets/250904_homograph_coordinate-plane2.jpg"

# RTSP 스트림 주소 (실시간 CCTV)
CAMERA_SOURCES = [
    "rtsp://210.99.70.120:1935/live/cctv001.stream",  # Camera 1
    "rtsp://210.99.70.120:1935/live/cctv001.stream",  # Camera 2
    "rtsp://210.99.70.120:1935/live/cctv001.stream",  # Camera 3
]

class Args:
    track_thresh = 0.3
    match_thresh = 0.9
    track_buffer = 180
    mot20 = False
    cpu_workers = 20
    chunk_sec = 5.0
    batch_size = 10

# =============================================================================
# 웹 서버 자동 실행
# =============================================================================
class WebServerManager:
    """웹 서버 자동 관리"""
    
    def __init__(self):
        self.web_process = None
        self.web_port = 8000
        self.web_url = f"http://localhost:{self.web_port}"
        
    def start_web_server(self):
        """웹 서버 백그라운드 실행"""
        try:
            print("🌐 웹 서버 시작 중...")
            
            # 웹 서버를 백그라운드에서 실행
            self.web_process = subprocess.Popen(
                [sys.executable, "app_web.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/workspace"
            )
            
            # 웹 서버 시작 대기
            time.sleep(3)
            
            # 웹 서버 상태 확인
            if self.web_process.poll() is None:
                print(f"✅ 웹 서버 시작 완료: {self.web_url}")
                return True
            else:
                print("❌ 웹 서버 시작 실패")
                return False
                
        except Exception as e:
            print(f"❌ 웹 서버 시작 오류: {e}")
            return False
    
    def open_browser(self):
        """웹 브라우저 자동 열기"""
        try:
            print("🌐 웹 브라우저 열기...")
            webbrowser.open(self.web_url)
            print(f"✅ 브라우저에서 {self.web_url} 열림")
        except Exception as e:
            print(f"❌ 브라우저 열기 실패: {e}")
            print(f"수동으로 {self.web_url}에 접속하세요")
    
    def stop_web_server(self):
        """웹 서버 종료"""
        if self.web_process and self.web_process.poll() is None:
            try:
                print("🌐 웹 서버 종료 중...")
                self.web_process.terminate()
                self.web_process.wait(timeout=5)
                print("✅ 웹 서버 종료 완료")
            except Exception as e:
                print(f"❌ 웹 서버 종료 오류: {e}")
                try:
                    self.web_process.kill()
                except:
                    pass

# =============================================================================
# 단일 모델 공유 시스템
# =============================================================================
class MultiCameraSystem:
    """멀티카메라 단일 모델 공유 시스템"""
    
    def __init__(self):
        self.shared_detector = None
        self.shared_tracker = None
        self.cameras = []
        self.engine = None
        self.viz = None
        self.streams = None
        self.web_manager = WebServerManager()
        
    def initialize_shared_models(self):
        """단일 모델 인스턴스 생성 (모든 카메라가 공유)"""
        print("📦 단일 모델 로드 중...")
        
        args = Args()
        det_models = ["vehicle"]
        
        self.shared_detector = DetectionAPI(
            models=det_models,
            thres=0.2,
            device="cuda:0",
            use_async=True,
            max_workers=1,
        )
        
        self.shared_tracker = TrackerAPI(args=args, detector=self.shared_detector)
        print("✅ 단일 모델 로드 완료")
        
        # GPU 메모리 사용량 확인
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"💾 GPU 메모리 사용량: {allocated:.2f}GB (단일 모델 공유)")
    
    def initialize_cameras(self):
        """카메라 초기화 (단일 모델 공유)"""
        print("📷 카메라 초기화 중...")
        print(f"🎥 카메라 소스: {CAMERA_SOURCES}")
        
        args = Args()
        plan_pts = [
            (1170, 214), (1170, 559), (1170, 904),
            (2212, 214), (2212, 559), (2212, 904),
            (3255, 214), (3255, 559), (3255, 904)
        ]
        
        # 각 카메라별 호모그래피 포인트
        camera_points = [
            [(1033,475), (948,474), (863,473), (1019,527), (890,524), (769,519), (973,667), (741,652), (548,620)],  # Camera 1
            [(518,466), (430,471), (341,474), (613,510), (485,519), (354,527), (829,608), (634,645), (397,668)],   # Camera 2
            [(357,602), (566,648), (832,683), (620,498), (754,509), (893,513), (726,453), (819,456), (911,459)]    # Camera 3
        ]
        
        self.cameras = []
        for i, (source, cam_pts) in enumerate(zip(CAMERA_SOURCES, camera_points)):
            try:
                cam = streamSCST(
                    source,
                    cam_pts,
                    PLAN_PATH,
                    plan_pts,
                    args,
                    detector=self.shared_detector,  # 공유 모델 주입
                    tracker=self.shared_tracker     # 공유 트래커 주입
                )
                self.cameras.append(cam)
                print(f"✅ Camera {i+1} 초기화 완료 (소스: {source})")
            except Exception as e:
                print(f"❌ Camera {i+1} 초기화 실패: {e}")
                # 실패한 카메라는 None으로 추가
                self.cameras.append(None)
        
        # 성공한 카메라만 필터링
        self.cameras = [cam for cam in self.cameras if cam is not None]
        print(f"🎯 총 {len(self.cameras)}개 카메라가 단일 모델을 공유합니다!")
    
    def initialize_visualization(self):
        """웹 시각화 초기화"""
        print("🎨 웹 시각화 초기화 중...")
        
        # 도면 파일 존재 확인
        if not os.path.exists(PLAN_PATH):
            print(f"❌ 도면 파일이 없습니다: {PLAN_PATH}")
            return False
            
        try:
            self.viz = WebPlanViz(plan_path=PLAN_PATH, show_cam_points=False, fps_limit=12.0)
            print("✅ WebPlanViz 초기화 완료")
        except Exception as e:
            print(f"❌ WebPlanViz 초기화 실패: {e}")
            return False
        
        # RTSP 스트림 (웹에서 원본 영상 보기용)
        print("📹 스트림 초기화 중...")
        self.streams = {}
        for i, source in enumerate(CAMERA_SOURCES):
            try:
                stream = CamMJPEG(name=f"cam{i+1}", url=source, width=480).start()
                self.streams[f"cam{i+1}"] = stream
                print(f"✅ Stream {i+1} 초기화 완료 (소스: {source})")
            except Exception as e:
                print(f"❌ Stream {i+1} 초기화 실패: {e}")
        
        # 웹 시각화와 스트림을 app_web.py에 연결
        set_webviz(self.viz)
        set_cam_streams(self.streams)
        print("✅ WebPlanViz가 웹 서버에 연결되었습니다")
        print(f"🔗 WebPlanViz 연결: {self.viz is not None}")
        print(f"✅ {len(self.streams)}개 카메라 스트림이 웹 서버에 연결되었습니다")
        print(f"🔗 카메라 스트림 연결: {len(self.streams)}개")
        print("✅ 웹 시각화와 스트림 연결 완료")
        
        return True
    
    def initialize_engine(self):
        """비동기 추론 엔진 초기화"""
        self.engine = AsyncEngine(
            self.cameras,
            interval=0.3,
            gpu_warning=90.0,      # 90% 이상: 프레임 스킵
            gpu_danger=95.0,       # 95% 이상: 추론 스킵
            gpu_critical=98.0,     # 98% 이상: 완전 정지
            gpu_recovery=85.0,     # 85% 이하: 정상 복구
            gpu_id=0
        )
    
    async def run(self):
        """메인 실행 루프 - 순서: 카메라 → 웹서버 → 추론"""
        print("🚀 멀티카메라 단일 모델 공유 시스템 시작")
        print(f"🎥 RTSP 스트림: {CAMERA_SOURCES[0]}")
        
        # 1단계: 카메라 초기화
        print("\n=== 1단계: 카메라 초기화 ===")
        self.initialize_shared_models()
        self.initialize_cameras()
        
        if not self.cameras:
            print("❌ 초기화된 카메라가 없습니다. 프로그램을 종료합니다.")
            return
        
        # 2단계: 웹 시각화 초기화
        print("\n=== 2단계: 웹 시각화 초기화 ===")
        if not self.initialize_visualization():
            print("❌ 웹 시각화 초기화 실패. 프로그램을 종료합니다.")
            return
        
        # 3단계: 웹 서버 시작
        print("\n=== 3단계: 웹 서버 시작 ===")
        if self.web_manager.start_web_server():
            # 브라우저 자동 열기
            self.web_manager.open_browser()
        else:
            print("⚠️ 웹 서버 시작 실패 - 웹 시각화 비활성화")
        
        # 4단계: 추론 엔진 초기화 및 실행
        print("\n=== 4단계: 추론 엔진 실행 ===")
        self.initialize_engine()
        
        # 스톨 감지용 변수
        last_seen = {"t": time.time(), "round": -1}
        
        async def watchdog(timeout_sec: int, stall_evt: asyncio.Event):
            while True:
                await asyncio.sleep(2)
                if time.time() - last_seen["t"] > timeout_sec:
                    logging.error("STALL detected → recycling all")
                    stall_evt.set()
        
        # 메인 루프
        while True:
            stall_evt = asyncio.Event()
            wd_task = asyncio.create_task(watchdog(20, stall_evt))
            
            try:
                async for result in self.engine.stream():
                    if stall_evt.is_set():
                        break
                    
                    last_seen["t"] = time.time()
                    last_seen["round"] = result["round"]
                    
                    # 모든 카메라의 좌표 수집
                    all_coords = []
                    for cam in result["cameras"]:
                        all_coords.extend(cam.get("plan_coords", []))
                    
                    # 웹 시각화 업데이트
                    if self.viz:
                        self.viz.update({
                            "round": result["round"],
                            "timestamp": result["timestamp"],
                            "fused": all_coords,
                            "cameras": result["cameras"],
                        })
                    
                    # 로깅
                    gpu_state = result.get("gpu_state", "UNKNOWN")
                    gpu_util = result.get("gpu_utilization", 0)
                    batch_time = result.get("batch_time", 0)
                    total_detections = result.get("total_detections", 0)
                    total_tracks = result.get("total_tracks", 0)
                    
                    logging.info(f"[SYSTEM] round={result['round']} objects={len(all_coords)} gpu={gpu_state}({gpu_util:.1f}%) batch={batch_time:.3f}s det={total_detections} track={total_tracks}")
                    
            except Exception as e:
                logging.error(f"[SYSTEM] recycling after error: {e}")
            finally:
                # 정리
                try:
                    self.engine.stop()
                except:
                    pass
                
                try:
                    wd_task.cancel()
                except:
                    pass
                
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
                
                await asyncio.sleep(2)
    
    def cleanup(self):
        """리소스 정리"""
        print("🧹 리소스 정리 중...")
        
        # 웹 서버 종료
        self.web_manager.stop_web_server()
        
        # 카메라 정리
        for cam in self.cameras:
            try:
                cam.close()
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
# 메인 실행
# =============================================================================
async def main():
    system = MultiCameraSystem()
    try:
        await system.run()
    except KeyboardInterrupt:
        print("\n중지됨")
    finally:
        system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
