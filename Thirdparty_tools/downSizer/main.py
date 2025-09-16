from projetion import *
from calibration.cam1 import CAM1_PTS
from calibration.cam2 import CAM2_PTS
from calibration.cam3 import CAM3_PTS
from calibration.cam1_resized import CAM1_PLAN_PTS_SCALED
from calibration.cam2_resized import CAM2_PLAN_PTS_SCALED
from calibration.cam3_resized import CAM3_PLAN_PTS_SCALED

PLAN_PTS = [CAM1_PTS, CAM2_PTS, CAM3_PTS]
PLAN_PTS_SCALED = [CAM1_PLAN_PTS_SCALED, CAM2_PLAN_PTS_SCALED, CAM3_PLAN_PTS_SCALED]


if __name__ == "__main__":
    # (A) 원본 도면 + 원본 좌표 → 자동 판별(또는 orig 명시)
    scaled = auto_project_points(
         image_path=r"D:\__________Workspace__________\video_cutter\downgradrer\asset\plan_origin.png",
         points=PLAN_PTS[0],
         save_path=r"D:\__________Workspace__________\video_cutter\downgradrer\project_origin.png",
         points_space="auto",           # 또는 "orig"        # 원본 도면 해상도
    )

    #(B) 축소 도면 + 축소 좌표 → 자동 판별(그대로 image로 처리)
    scaled = auto_project_points(
        image_path=r"D:\__________Workspace__________\video_cutter\downgradrer\asset\plan_resized.png",
        points=PLAN_PTS_SCALED[0],
        save_path=r"D:\__________Workspace__________\video_cutter\downgradrer\project_downsize.png",
        points_space="auto",  )         # 좌표가 이미지 범위 이내면 그대로 사용
