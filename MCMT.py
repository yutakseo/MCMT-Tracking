# /workspace/main.py
import logging, os, asyncio
import numpy as np
from MCMT_engine.SCST import SCST
from MCMT_engine.async_inference import AsyncEngine  # AsyncEngine 모듈 경로에 맞게 조정
from MCMT_engine.visualizer import PlanVisualizer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|buffer_size;102400|max_delay;0|stimeout;5000000"
)

class Args:
    track_thresh: float = 0.5
    match_thresh: float = 0.5
    track_buffer: int = 60
    mot20: bool = False
    cpu_workers: int = 10

def build_cameras():
    args = Args()
    rtsp = "rtsp://210.99.70.120:1935/live/cctv001.stream"
    plan_path = "/workspace/assets/25082_homograph_coordinate-plane.jpg"
    plan_pts  = [
        (3588, 412), (3588, 1036), (3588, 1657),
        (2225, 412), (2225, 1036), (2225, 1657),
        (861, 412),  (861, 1036),  (861, 1657),
    ]
    cam1_pts = [(1390,521),(1618,552),(1784,578),(1112,564),(1434,620),(1709,668),(481,651),(852,809),(1393,1007)]
    cam2_pts = [(1020,1027),(476,898),(294,826),(1348,790),(1017,772),(813,753),(1437,716),(1213,713),(1058,707)]
    cam3_pts = [(140,309),(291,315),(398,320),(213,376),(440,377),(588,378),(467,596),(826,521),(965,484)]
    return [
        SCST(rtsp, cam1_pts, plan_path, plan_pts, args),
        SCST(rtsp, cam2_pts, plan_path, plan_pts, args),
        SCST(rtsp, cam3_pts, plan_path, plan_pts, args),
    ]

async def main():
    cams = build_cameras()
    engine = AsyncEngine(cams, history_len=20)

    viz = PlanVisualizer(
        cams,
        plan_path="/workspace/assets/25082_homograph_coordinate-plane.jpg",  # projector에 플랜 없으면 여기서 로드
        show_window=True,
        draw_cam_points=False,                 # 원시 카메라 포인트도 보려면 True
        video_path=None,                       # "/workspace/fusion.mp4" 등으로 지정하면 저장
        video_fps=10,
    )

    try:
        async for pkt in engine.stream():
            # 시각화
            canvas, quit_flag = viz.render(pkt)
            # 실시간으로 fused 값 활용
            fused = pkt.get("fused", [])
            print(f"[APP] round={pkt['round']} fused_n={len(fused)} preview={fused[:5]}")
            # 예: 좌표 결과를 다른 시스템으로 전송/저장/가공
            # send_to_kafka(fused)  # <- 예시

            if quit_flag:
                break
    finally:
        viz.close()




if __name__ == "__main__":
    asyncio.run(main())
