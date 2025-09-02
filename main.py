# /workspace/main.py
import os, logging
from tools.SCST import SCST
from async_inference import AsyncEngine

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")

# RTSP 옵션(선택)
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
    cam1 = SCST(rtsp, cam1_pts, plan_path, plan_pts, args)
    cam2 = SCST(rtsp, cam2_pts, plan_path, plan_pts, args)
    cam3 = SCST(rtsp, cam3_pts, plan_path, plan_pts, args)
    return [cam1, cam2, cam3]

def main():
    cams = build_cameras()
    engine = AsyncEngine(cams, history_len=10)

    engine.run_sync(max_rounds=None)  
    print("Result :", engine.result)              
    print("History:", engine.result_history)  


    # pkts = engine.run_sync(max_rounds=20)             
    # print("latest(result):", engine.result)
    # print("history len:", len(engine.result_history))

if __name__ == "__main__":
    main()
