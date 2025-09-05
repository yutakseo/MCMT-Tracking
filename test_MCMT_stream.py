# /workspace/test_main.py
import os
import logging
import asyncio
import threading

from MCMT_engine.stream_SCST import streamSCST  # SCST -> streamSCST로 변경
from MCMT_engine.async_inference import AsyncEngine
from tools.webviz import WebPlanViz
from app_web import serve_webviz

# ← RTSP MJPEG용 (CCTVStreamer 우선, 없으면 OpenCV 폴백)
from MCMT_engine.cam_stream import CamMJPEG

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

# FFmpeg/RTSP 옵션(지연↓, 타임아웃 설정)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|buffer_size;102400|max_delay;0|stimeout;5000000"
)

# ─────────────────────────────────────────────────────────────────────────────
# 구성
# ─────────────────────────────────────────────────────────────────────────────
PLAN_PATH = "/workspace/assets/250904_homograph_coordinate-plane2.jpg"
RTSP1_URL  = "rtsp://210.99.70.120:1935/live/cctv001.stream"
RTSP2_URL  = "rtsp://210.99.70.120:1935/live/cctv001.stream"
RTSP3_URL  = "rtsp://210.99.70.120:1935/live/cctv001.stream"

class Args:
    track_thresh = 0.3
    match_thresh = 0.9
    track_buffer = 180
    mot20 = False
    cpu_workers = 20   # 듀얼 CPU 적극 활용
    chunk_sec   = 10.0 # 15~30초 권장
    batch_size = 20

def build_cameras():
    args = Args()
    det_models = ["vehicle"]
    plan_pts  = [
                (1170,  214),    #point1
                (1170,  559),   #point2
                (1170,  904),   #point3
                (2212,  214),    #point4
                (2212,  559),   #point5
                (2212,  904),   #point6
                (3255,  214),     #point7
                (3255,  559),    #point8
                (3255,  904),    #point9
                ]
    
    cam1_pts = [(1033,475),
                (948,474),
                (863,473),
                (1019,527),
                (890,524),
                (769,519),
                (973,667),
                (741,652),
                (548,620)]
    
    cam2_pts = [(518,466),
                (430,471),
                (341,474),
                (613,510),
                (485,519),
                (354,527),
                (829,608), 
                (634,645),
                (397,668)]
    
    cam3_pts = [(357,602),
                (566,648),
                (832,683),
                (620,498),
                (754,509),
                (893,513),
                (726,453),
                (819,456),
                (911,459)]

    return [
        streamSCST(RTSP1_URL, cam1_pts, PLAN_PATH, plan_pts, args, det_models=det_models),  # SCST -> streamSCST
        streamSCST(RTSP2_URL, cam2_pts, PLAN_PATH, plan_pts, args, det_models=det_models),  # SCST -> streamSCST
        streamSCST(RTSP3_URL, cam3_pts, PLAN_PATH, plan_pts, args, det_models=det_models),  # SCST -> streamSCST
    ]

def build_rtsp_streams():
    """
    웹에서 볼 '원본 RTSP' 스트림들 (CCTVStreamer → MJPEG 변환).
    CCTVStreamer(pybind11)가 있으면 자동 사용, 없으면 OpenCV VideoCapture 폴백.
    """
    streams = {
        "cam1": CamMJPEG(name="cam1", url=RTSP1_URL, width=480).start(),
        "cam2": CamMJPEG(name="cam2", url=RTSP2_URL, width=480).start(),
        "cam3": CamMJPEG(name="cam3", url=RTSP3_URL, width=480).start()
        # 서로 다른 RTSP라면 각각 다른 URL로 지정하면 됩니다.
    }
    return streams

# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    cams = build_cameras()
    engine = AsyncEngine(cams, interval=0.1)  # history_len 제거, interval 추가

    # 웹 비주얼라이저 (도면 위 fused 좌표 시각화)
    viz = WebPlanViz(
        plan_path=PLAN_PATH,
        show_cam_points=False,   # 필요시 True로
        fps_limit=15.0
    )

    # RTSP 원본도 웹으로 송출
    streams = build_rtsp_streams()

    # FastAPI 웹서버를 백그라운드에서 실행
    t = threading.Thread(target=serve_webviz, args=(viz,), kwargs={"streams": streams, "host": "0.0.0.0", "port": 8000}, daemon=True)
    t.start()

    # 스트리밍 루프: 라운드마다 패킷을 받고 웹 프레임 갱신
    async for result in engine.stream():  # pkt -> result로 변경
        # 새로운 결과 구조에 맞게 수정
        timestamp = result['timestamp']
        round_num = result['round']
        cameras = result['cameras']
        
        # 모든 카메라의 도면 좌표 수집
        all_coords = []
        for cam_data in cameras:
            all_coords.extend(cam_data['plan_coords'])
        
        # 웹 비주얼라이저 업데이트 (기존 pkt 구조로 변환)
        pkt = {
            'round': round_num,
            'timestamp': timestamp,
            'fused': all_coords,  # 모든 카메라의 좌표를 fused로 사용
            'cameras': cameras
        }
        
        viz.update(pkt)  # 도면 프레임 갱신
        logging.info(f"[APP] round={round_num} total_objects={len(all_coords)} preview={all_coords[:3]}")

if __name__ == "__main__":
    asyncio.run(main()) 