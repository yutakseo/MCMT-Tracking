from projetion import *
from calibration.cam1 import CAM1_PTS
from calibration.cam2 import CAM2_PTS
from calibration.cam3 import CAM3_PTS
from calibration.cam1_resized import CAM1_PLAN_PTS_SCALED
from calibration.cam2_resized import CAM2_PLAN_PTS_SCALED
from calibration.cam3_resized import CAM3_PLAN_PTS_SCALED

PLAN_PTS = [CAM1_PTS, CAM2_PTS, CAM3_PTS]
PLAN_PTS_SCALED = [CAM1_PLAN_PTS_SCALED, CAM2_PLAN_PTS_SCALED, CAM3_PLAN_PTS_SCALED]



print(len(CAM2_PTS))
print(len(CAM2_PLAN_PTS_SCALED))