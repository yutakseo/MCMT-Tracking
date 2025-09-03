# /workspace/test_main.py
import os
import logging
import asyncio
import threading

from tools.SCST import SCST
from tools.async_inference import AsyncEngine
from tools.webviz import WebPlanViz
from app_web import serve_webviz

# ← RTSP MJPEG용 (CCTVStreamer 우선, 없으면 OpenCV 폴백)
from tools.cam_stream import CamMJPEG

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

# FFmpeg/RTSP 옵션(지연↓, 타임아웃 설정)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|buffer_size;102400|max_delay;0|stimeout;5000000"
)

# ─────────────────────────────────────────────────────────────────────────────
# 구성
# ─────────────────────────────────────────────────────────────────────────────
PLAN_PATH = "/workspace/assets/250903_homograph_coordinate-plane2.jpg"
RTSP1_URL  = "rtsp://admin:asdf2510$_@192.168.50.131:554/video1"
RTSP2_URL  = "rtsp://admin:asdf2510$_@192.168.50.131:554/video2"
RTSP3_URL  = "rtsp://admin:asdf2510$_@192.168.50.131:554/video3"

class Args:
    track_thresh: float = 0.5
    match_thresh: float = 0.5
    track_buffer: int = 60
    mot20: bool = False
    cpu_workers: int = 10

def build_cameras():
    args = Args()
    plan_pts  = [
                ( 792,  504),    #point1
                ( 792,  871),   #point2
                ( 792, 1250),   #point3
                (2112,  504),    #point4
                (2112,  871),   #point5
                (2112, 1250),   #point6
                (3432,  504),     #point7
                (3432,  871),    #point8
                (3432, 1250),    #point9
                ]
    
    cam1_pts = [(1390,521),(1618,552),(1784,578),(1112,564),(1434,620),(1709,668),(481,651),(852,809),(1393,1007)]
    cam2_pts = [(1020,1027),(476,898),(294,826),(1348,790),(1017,772),(813,753),(1437,716),(1213,713),(1058,707)]
    cam3_pts = [(140,309),(291,315),(398,320),(213,376),(440,377),(588,378),(467,596),(826,521),(965,484)]

    return [
        SCST(RTSP1_URL, cam1_pts, PLAN_PATH, plan_pts, args),
        SCST(RTSP2_URL, cam2_pts, PLAN_PATH, plan_pts, args),
        SCST(RTSP3_URL, cam3_pts, PLAN_PATH, plan_pts, args),
    ]

def build_rtsp_streams():
    """
    웹에서 볼 ‘원본 RTSP’ 스트림들 (CCTVStreamer → MJPEG 변환).
    CCTVStreamer(pybind11)가 있으면 자동 사용, 없으면 OpenCV VideoCapture 폴백.
    """
    streams = {
        "cam1": CamMJPEG(name="cam1", url=RTSP_URL, width=1080).start()
        # 서로 다른 RTSP라면 각각 다른 URL로 지정하면 됩니다.
    }
    return streams

# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
async def main():
    cams = build_cameras()
    engine = AsyncEngine(cams, history_len=20)

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
    async for pkt in engine.stream():
        fused = pkt.get("fused", [])
        viz.update(pkt)  # 도면 프레임 갱신
        logging.info(f"[APP] round={pkt['round']} fused_n={len(fused)} preview={fused[:3]}")

if __name__ == "__main__":
    asyncio.run(main())
