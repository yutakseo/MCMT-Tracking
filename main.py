#/workspace/main.py
from camera import Camera

# ----------------------------
# Args
# ----------------------------
class Args:
    track_thresh: float = 0.5
    match_thresh: float = 0.5
    track_buffer: int = 60
    mot20: bool = False
    cpu_workers: int = 10



# ----------------------------
# Example
# ----------------------------
if __name__ == "__main__":
    args = Args()

    # (예시) 좌표: 반드시 1:1 대응 & 일관된 순서
    cam_pts = [(1390, 521), (1618, 552), (1784, 578), (1112, 564)]
    plan_pts = [(100, 200), (300, 210), (320, 400), (90, 380)]

    cam = Camera(
        track_args=args,
        cctv_url="rtsp://210.99.70.120:1935/live/cctv001.stream",
        cord_plan="/workspace/assets/25082_homograph_coordinate-plane.jpg",
        plan_benchmark=plan_pts,
        cam_pts=cam_pts,
        results_dir="./results",
    )

    # 방법 1) 단순 스트리밍 + 종료 시 마지막 좌표만 받기
    positions = cam.stream_plan(return_positions=True, mode="bottom-center")
    print("마지막 프레임 객체 좌표:", positions)
