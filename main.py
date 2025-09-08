# /workspace/main.py - 멀티카메라 단일 모델 공유 시스템 (웹 자동 실행)

import os
import logging
import time
import asyncio
import sys
import threading
import webbrowser
import socket
import urllib.request
from pathlib import Path

sys.path.append("/workspace")

from mmyolo.utils import register_all_modules
register_all_modules()

from MCMT_engine.stream_SCST import streamSCST
from MCMT_engine.async_inference import AsyncEngine
from tools.webviz import WebPlanViz
from __Detection.detection_api import DetectionAPI
from __Tracking.tracking_api import TrackerAPI
from MCMT_engine.cam_stream import CamMJPEG

# 같은 프로세스/스레드에서 uvicorn 실행을 위해 app과 setter를 가져옴
import uvicorn
from app_web import app, set_webviz, set_cam_streams, set_cam_overlays

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
# 웹 서버 자동 실행 (같은 프로세스/스레드)
# =============================================================================
class WebServerManager:
    """웹 서버 자동 관리 (동일 프로세스/스레드)"""

    def __init__(self):
        self.web_thread = None
        self.web_port = 8000
        self.web_url = f"http://localhost:{self.web_port}"

    def _port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex(("127.0.0.1", port)) == 0

    def _probe_healthz(self, url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=0.6) as r:
                return r.read(2) == b"ok"
        except Exception:
            return False

    def _find_free_port(self, start=8000, end=8100) -> int:
        for p in range(start, end + 1):
            if not self._port_in_use(p):
                return p
        raise RuntimeError("No free port found in range 8000-8100")

    def start_web_server(self):
        """uvicorn을 같은 프로세스에서 스레드로 실행. 이미 떠 있으면 재사용."""
        try:
            # 1) 기존 8000 살아있으면 재사용
            if self._port_in_use(self.web_port):
                if self._probe_healthz(f"http://127.0.0.1:{self.web_port}/healthz"):
                    print(f"기존 웹 서버 재사용: http://localhost:{self.web_port}")
                    self.web_url = f"http://localhost:{self.web_port}"
                    return True
                else:
                    # 2) 헬스체크 실패 → 빈 포트로 이동
                    new_port = self._find_free_port()
                    print(f"8000 사용중(헬스 실패). 포트를 {new_port}로 변경합니다.")
                    self.web_port = new_port
                    self.web_url = f"http://localhost:{self.web_port}"

            print("웹 서버(동일 프로세스 스레드) 시작 중...")
            def _run():
                uvicorn.run(app, host="0.0.0.0", port=self.web_port, log_level="info")
            self.web_thread = threading.Thread(target=_run, daemon=True)
            self.web_thread.start()

            # 부팅 대기 및 헬스체크
            for _ in range(40):
                if self._probe_healthz(f"http://127.0.0.1:{self.web_port}/healthz"):
                    print(f"웹 서버 시작 완료: {self.web_url}")
                    return True
                time.sleep(0.2)

            print("웹 서버 시작 확인 실패(헬스체크 타임아웃)")
            return False

        except Exception as e:
            print(f"웹 서버 시작 오류: {e}")
            return False

    def open_browser(self):
        try:
            print("웹 브라우저 열기...")
            webbrowser.open(self.web_url)
            print(f"브라우저에서 {self.web_url} 열림")
        except Exception as e:
            print(f"브라우저 열기 실패: {e} → 수동 접속: {self.web_url}")

    def stop_web_server(self):
        print("uvicorn 스레드는 프로세스 종료 시 함께 정리됩니다.")

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
        print("단일 모델 로드 중...")
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
        print("단일 모델 로드 완료")

        # GPU 메모리 사용량 확인
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"GPU 메모리 사용량: {allocated:.2f}GB (단일 모델 공유)")

    def initialize_cameras(self):
        """카메라 초기화 (단일 모델 공유)"""
        print("카메라 초기화 중...")
        print(f"카메라 소스: {CAMERA_SOURCES}")

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
                print(f"Camera {i+1} 초기화 완료 (소스: {source})")
            except Exception as e:
                print(f"Camera {i+1} 초기화 실패: {e}")
                self.cameras.append(None)

        # 성공한 카메라만 필터링
        self.cameras = [cam for cam in self.cameras if cam is not None]
        print(f"총 {len(self.cameras)}개 카메라가 단일 모델을 공유합니다!")

    def initialize_visualization(self):
        """웹 시각화 초기화"""
        print("웹 시각화 초기화 중...")

        # 도면 파일 존재 확인
        if not os.path.exists(PLAN_PATH):
            print(f"도면 파일이 없습니다: {PLAN_PATH}")
            return False

        try:
            self.viz = WebPlanViz(plan_path=PLAN_PATH, show_cam_points=False, fps_limit=12.0)
            print("WebPlanViz 초기화 완료")
        except Exception as e:
            print(f"WebPlanViz 초기화 실패: {e}")
            return False

        # RTSP 스트림 (웹에서 원본 영상 보기용)
        print("스트림 초기화 중...")
        self.streams = {}
        for i, source in enumerate(CAMERA_SOURCES):
            try:
                print(f"Stream {i+1} 연결 시도 중... (소스: {source})")

                stream = CamMJPEG(
                    name=f"cam{i+1}",
                    url=source,
                    width=480,
                    jpeg_quality=85
                )
                stream = stream.start()

                # 연결 테스트 (3초 대기)
                print(f"Stream {i+1} 연결 테스트 중...")
                time.sleep(3)

                if getattr(stream, "_jpg", None) and len(stream._jpg) > 1000:
                    self.streams[f"cam{i+1}"] = stream
                    print(f"Stream {i+1} 초기화 완료 (소스: {source})")
                else:
                    print(f"Stream {i+1} 연결 불안정 - 재시도 중...")
                    stream.stop()
                    time.sleep(2)
                    stream = CamMJPEG(name=f"cam{i+1}", url=source, width=480, jpeg_quality=85).start()
                    time.sleep(3)
                    if getattr(stream, "_jpg", None) and len(stream._jpg) > 1000:
                        self.streams[f"cam{i+1}"] = stream
                        print(f"Stream {i+1} 재연결 성공")
                    else:
                        print(f"Stream {i+1} 연결 실패 - 건너뜀")

            except Exception as e:
                print(f"Stream {i+1} 초기화 실패: {e}")
                # 실패한 스트림은 테스트 영상으로 대체 (옵션)
                try:
                    print(f"Stream {i+1} 테스트 영상으로 대체...")
                    test_stream = self._create_test_stream(f"cam{i+1}")
                    if test_stream:
                        self.streams[f"cam{i+1}"] = test_stream
                        print(f"Stream {i+1} 테스트 영상으로 대체 완료")
                except Exception as test_e:
                    print(f"Stream {i+1} 테스트 영상 생성 실패: {test_e}")

        # 웹 시각화와 스트림을 app_web.py에 연결 (같은 프로세스 전역에 주입)
        set_webviz(self.viz)
        set_cam_streams(self.streams)
        print(f"WebPlanViz 연결: {self.viz is not None}")
        print(f"{len(self.streams)}개 카메라 스트림이 웹 서버에 연결되었습니다")
        print("웹 시각화와 스트림 연결 완료")

        return True

    def _create_test_stream(self, name: str):
        """테스트용 스트림 생성 (RTSP 연결 실패 시 대체)"""
        try:
            import cv2
            import numpy as np

            class TestStream:
                def __init__(self, name):
                    self.name = name
                    self._jpg = None
                    self._counter = 0
                    self._lock = threading.Lock()
                    self._stop = False
                    self._th = threading.Thread(target=self._test_loop, daemon=True)
                    self._th.start()

                def _test_loop(self):
                    while not self._stop:
                        try:
                            h, w = 480, 640
                            xs = np.linspace(0, 1, w, dtype=np.float32)
                            ys = np.linspace(0, 1, h, dtype=np.float32)
                            grad_x = (xs * 255).astype(np.uint8)[None, :].repeat(h, axis=0)
                            grad_y = (ys * 255).astype(np.uint8)[:, None].repeat(w, axis=1)
                            r_val = int((0.5 + 0.5*np.sin(self._counter*0.1)) * 255)
                            r = np.full_like(grad_x, r_val)
                            img = np.dstack([grad_x, grad_y, r])

                            cv2.putText(img, f"TEST STREAM: {self.name}", (50, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(img, "RTSP Connection Failed", (50, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                            cv2.putText(img, f"Time: {time.strftime('%H:%M:%S')}", (50, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                            cx = int(320 + 200 * np.sin(self._counter * 0.1))
                            cy = int(240 + 100 * np.cos(self._counter * 0.1))
                            cv2.circle(img, (cx, cy), 30, (255, 255, 0), -1)

                            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])

                            with self._lock:
                                self._jpg = buffer.tobytes()

                            self._counter += 1
                            time.sleep(0.1)  # ~10 FPS

                        except Exception as e:
                            print(f"Test stream error: {e}")
                            time.sleep(1)

                def stop(self):
                    self._stop = True
                    if self._th and self._th.is_alive():
                        self._th.join(timeout=1.0)

            return TestStream(name)

        except Exception as e:
            print(f"Test stream creation failed: {e}")
            return None

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
        print("멀티카메라 단일 모델 공유 시스템 시작")
        print(f"RTSP 스트림: {CAMERA_SOURCES[0]}")

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

        # 3단계: 웹 서버 시작 (같은 프로세스/스레드)
        print("\n=== 3단계: 웹 서버 시작 ===")
        if self.web_manager.start_web_server():
            try:
                self.web_manager.open_browser()  # 선택
            except Exception:
                pass
        else:
            print("❌ 웹 서버 시작 실패 - 웹 시각화 비활성화")

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

                    # 모든 카메라의 좌표/오버레이 수집
                    all_coords = []
                    coords_per_cam = []
                    overlays = {}

                    for idx, cam in enumerate(result["cameras"]):
                        # plan 좌표
                        pts = cam.get("plan_coords", [])
                        all_coords.extend(pts)
                        coords_per_cam.append(pts)

                        # 오버레이 (tracks 우선 → detections 보조)
                        items = []
                        for t in (cam.get("tracks") or []):
                            bbox = t.get("bbox") or t.get("tlbr") or t.get("xyxy")
                            if bbox and len(bbox) >= 4:
                                items.append({
                                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                                    "label": t.get("label") or t.get("cls_name") or "obj",
                                    "score": float(t.get("score", 1.0)),
                                    "track_id": t.get("track_id") or t.get("id")
                                })
                        if not items:
                            for d in (cam.get("detections") or []):
                                bbox = d.get("bbox") or d.get("tlbr") or d.get("xyxy")
                                if bbox and len(bbox) >= 4:
                                    items.append({
                                        "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                                        "label": d.get("label") or d.get("cls_name") or "obj",
                                        "score": float(d.get("score", 1.0)),
                                        "track_id": None
                                    })
                        overlays[f"cam{idx+1}"] = items

                    # 웹 시각화 업데이트 (키 이름: coords)
                    if self.viz:
                        self.viz.update({
                            "round": result["round"],
                            "timestamp": result["timestamp"],
                            "fused": all_coords,
                            "coords": coords_per_cam,
                        })

                    # 카메라 뷰 오버레이 주입
                    set_cam_overlays(overlays)

                    # 로깅
                    gpu_state = result.get("gpu_state", "UNKNOWN")
                    gpu_util = result.get("gpu_utilization", 0)
                    batch_time = result.get("batch_time", 0)
                    total_detections = result.get("total_detections", 0)
                    total_tracks = result.get("total_tracks", 0)

                    logging.info(
                        f"[SYSTEM] round={result['round']} objects={len(all_coords)} "
                        f"gpu={gpu_state}({gpu_util:.1f}%) batch={batch_time:.3f}s "
                        f"det={total_detections} track={total_tracks}"
                    )

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
        print("리소스 정리 중...")

        # 웹 서버 종료 (동일 프로세스 스레드는 프로세스 종료 시 함께 종료)
        self.web_manager.stop_web_server()

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

        print("정리 완료")

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
