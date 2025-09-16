# resize_points_and_save.py
from typing import List, Tuple
import os

Point = Tuple[int, int]

# === 원본 좌표 (1920×1080 기준) ===
CAM1_PTS = [
    (371, 513),
    (251, 445),
    (414, 373),
    (312, 336),
    (114, 286),  # 메인호수
    (70, 332),
    (249, 315),
    (159, 256),  # 미니저수지1
    (176, 259),  # 미니저수지2
    (212, 264),  # 미니저수지3
    (628, 352),
    (874, 323),  # 노란장비
    (748, 310),
    (464, 264),
    (327, 243),  # 출입구 잘안보이는 좌표
    (371, 237),
    (404, 235),
    (442, 229),
    (516, 235),  # 출입구 잘안보이는 좌표
    (594, 253),  # 방음벽 잘안보이는 좌표
    (640, 258),
    (662, 254),
    (740, 262),
    (829, 268),  # 방음벽 잘안보이는 좌표
    (1065, 305)  # 타워크레인
]

# === 스케일링 함수 ===
def scale_points(points: List[Point], src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> List[Point]:
    """
    원본 해상도(src_size) → 목표 해상도(dst_size)로 좌표 스케일링
    src_size, dst_size: (width, height)
    """
    scale_x = dst_size[0] / src_size[0]
    scale_y = dst_size[1] / src_size[1]
    return [(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in points]


if __name__ == "__main__":
    # 해상도 설정 (1920x1080 → 1280x720)
    src_size = (1920, 1080)
    dst_size = (1280, 720)

    # 좌표 변환
    scaled_pts = scale_points(CAM1_PTS, src_size, dst_size)

    # 저장 경로
    save_path ="/workspace/Thirdparty_tools/resized.py"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 파일 저장
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("# 자동 생성된 1920x1080 → 1280x720 변환 좌표\n")
        f.write("CAM1_PLAN_PTS_SCALED = [\n")
        for x, y in scaled_pts:
            f.write(f"    ({x}, {y}),\n")
        f.write("]\n")

    print(f"✅ 좌표 저장 완료 → {save_path}")
    print(f"총 {len(scaled_pts)}개 포인트 저장됨.")
