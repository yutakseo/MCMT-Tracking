# /workspace/main_video.py
# 단일 videoSCST 인스턴스를 재사용하여 카메라 #1 → #2 → #3 순차 처리
# - 카메라별 플랜(도면) 기준점 개수가 서로 달라도 OK (호출마다 plan_pts 오버라이드)
# - 주석/줄바꿈 그대로 유지한 좌표 블록을 아래에 붙여 넣어 사용

from __future__ import annotations
import sys
from multiprocessing import freeze_support

sys.path.append("/workspace")

from MCMT_engine.SCST.video_SCST import videoSCST, Args


# ─────────────────────────────────────────────────────────
# 사용자 설정
# ─────────────────────────────────────────────────────────
class VidArgs(Args):
    track_thresh = 0.3
    match_thresh = 0.9
    track_buffer = 180
    mot20 = False
    cpu_workers = 20    # 노트북/디버거면 1 권장
    chunk_sec   = 10.0
    batch_size  = 20

# (1) 도면 이미지(공통)
PLAN_IMG_PATH = "/workspace/assets/seocho/Seocho_plan_pts.png"

# (2) 카메라별 '플랜(도면) 기준점'  ── 주석/줄바꿈 유지
cam1_plan_pts = [
    (1431,1198), #측정안전통로
    (1505,1256), #노란깃발
    (1486,1068), #철근아래
    (1517,999),  #철근위
    (1505,972),  #나무적재왼쪽
    (1528,972),  #나무적재오른쪽
    (1601,1015), #범퍼
    (1664,1015), #초록깃발
    (1699,958),  #미니저수지왼쪽코너
    (1724,644),  #미니저수지위쪽코너
    (2026,449),  #출입구왼쪽
    (2032,532),  #출입구오른쪽
    (2016,615),  #컨테이너왼쪽
    (2019,654),  #컨테이너오른쪽
    (1917,939),  #미니저수지오른쪽
    (1908,625),  #미니저수지오른쪽윗코너
    (1219,1557), #빨간꼬깔
]

cam2_plan_pts = [
    (1597,625),  #저수지 계단 아래
    (1629,619),  #저수지 계단 위
    (2023,451),  #출입구 왼쪽
    (2029,530),  #출입구 오른쪽
    (2015,615),  #컨테이너 왼쪽
    (2019,659),  #컨테이너 오른쪽
    (1372,907),  #저수지 꼭지
    (1375,952),  #꼭지 밑 안전통로
    (1505,973),  #나무적재왼쪽
    (1525,971),  #나무적재오른쪽
    (1520,998),  #철근위
    (1485,1068), #철근아래
    (1330,1177), #안전통로입구
    (1431,1198), #측정안전통로
    (1594,1176), #범퍼
    (1504,1253), #노란깃발
    (1698,960),  #작은저수지왼쪽코너
    (1915,942),  #작은저수지오른쪽코너
]

cam3_plan_pts = [
    (1597,627),  #저수지계단아래
    (1635,619),  #저수지계단위
    (1505,1256), #노란깃발
    (1695,958),  #미니저수지왼쪽아래
    (1663,1017), #초록깃발
    (1036,378),  #미니저수지오른쪽아래
    (1724,643),  #미니저수지왼쪽위
    (1910,630),  #미니저수지오른쪽 위
    (1597,1174), #범퍼
    (1616,903),  #파란깃발
    (2025,453),  #출입구왼쪽
    (2028,529),  #출입구오른쪽
    (2015,614),  #컨테이너왼쪽
    (2019,662),  #컨테이너 오른쪽
    (769,587),   #파란포장
    (1505,973),  #나무적재왼쪽
    (1531,967),  #나무적재오른쪽
    (1477,1088), #철근아래
    (1454,411),  #직원휴게실왼쪽
    (1568,377),  #직원휴게실오른쪽
]

# (3) 카메라별 'CCTV(영상) 기준점'  ── 주석/줄바꿈 유지
cam1_pts = [
    (342,661),  #측정안전통로
    (733,657),  #노란깃발
    (266,567),  #철근아래
    (230,526),  #철근위
    (129,519),  #나무적재왼쪽
    (188,514),  #나무적재오른쪽
    (811,603),  #범퍼
    (612,525),  #초록깃발
    (600,516),  #미니저수지왼쪽코너
    (177,403),  #미니저수지위쪽코너
    (420,382),  #출입구왼쪽
    (506,395),  #출입구오른쪽
    (577,421),  #컨테이너왼쪽
    (623,416),  #컨테이너오른쪽
    (979,493),  #미니저수지오른쪽
    (451,418),  #미니저수지오른쪽윗코너
    (961,717),  #빨간꼬깔
]

cam2_pts = [
    (2,462),    #저수지 계단아래
    (25,420),   #저수지 계단 위
    (107,391),  #출입구 왼쪽
    (212,399),  #출입구 오른쪽
    (309,409),  #컨테이너 왼쪽
    (360,417),  #컨테이너 오른쪽
    (313,542),  #저수지 꼭지
    (319,564),  #꼭지 밑 안전통로
    (561,503),  #나무적재왼쪽
    (580,496),  #나무적재오른쪽
    (633,506),  #철근위
    (782,530),  #철근 아래
    (966,608),  #안전통로입구
    (1055,571), #측정안전통로
    (1051,531), #범퍼
    (1206,555), #노란깃발
    (674,485),  #작은저수지왼쪽코너
    (728,459),  #작은저수지오른쪽코너
]

cam3_pts = [
    (267,350),  #저수지계단아래
    (310,305),  #저수지계단위
    (104,706),  #노란깃발
    (649,407),  #미니저수지왼쪽아래
    (664,422),  #초록깃발
    (1036,378), #미니저수지오른쪽아래
    (448,309),  #미니저수지왼쪽위
    (692,318),  #미니저수지오른쪽 위
    (539,549),  #범퍼
    (372,362),  #파란깃발
    (711,299),  #출입구왼쪽
    (763,306),  #출입구오른쪽
    (798,311),  #컨테이너왼쪽
    (827,319),  #컨테이너 오른쪽
    (769,587),  #파란포장
    (119,419),  #나무적재왼쪽
    (176,408),  #나무적재오른쪽
    (60,471),   #철근아래
    (63,301),   #직원휴게실왼쪽
    (192,291),  #직원휴게실오른쪽
]

# (4) 입력 비디오 경로(서초 3카메라)
VIDEO1 = "/workspace/datasets/250909_Site_seocho/2025-09-09 13_29_59 이동형 #1.mp4"
VIDEO2 = "/workspace/datasets/250909_Site_seocho/2025-09-09 13_29_59 이동형 #2.mp4"
VIDEO3 = "/workspace/datasets/250909_Site_seocho/2025-09-09 13_29_59 이동형 #3.mp4"

# (5) 결과 저장 경로
OUT1_CAM = "/workspace/results/tracking_result1.mp4"
OUT1_MAP = "/workspace/results/plan_result1.mp4"

OUT2_CAM = "/workspace/results/tracking_result2.mp4"
OUT2_MAP = "/workspace/results/plan_result2.mp4"

OUT3_CAM = "/workspace/results/tracking_result3.mp4"
OUT3_MAP = "/workspace/results/plan_result3.mp4"

# (6) 사용 모델
MODELS = ["ultra_people", "worker"]


def main():
    args = VidArgs()

    # 노트북/디버거/Windows spawn 이슈 있으면 ↓ 1로 낮추고 돌리세요.
    # args.cpu_workers = 1

    # 인스턴스 1회 생성(Detector/Tracker 재사용) → 카메라별로 H만 다시 맞춰 처리
    scst = videoSCST(
        args=args,
        plan_img_path=PLAN_IMG_PATH,
        plan_pts=cam1_plan_pts,         # 초기값(이후 호출에서 plan_pts로 덮어씀)
        det_models=MODELS,
    )

    # Cam #1
    scst.track_and_save(
        video_path=VIDEO1,
        cctv_pts=cam1_pts,              # CCTV 기준점
        plan_pts=cam1_plan_pts,         # 플랜 기준점 (호출마다 override)
        plan_img_path=PLAN_IMG_PATH,
        camera_save_path=OUT1_CAM,
        plan_save_path=OUT1_MAP,
        plan_mode="bottom-center",
        cam_trail_len=30,
    )

    # Cam #2
    scst.track_and_save(
        video_path=VIDEO2,
        cctv_pts=cam2_pts,
        plan_pts=cam2_plan_pts,
        plan_img_path=PLAN_IMG_PATH,
        camera_save_path=OUT2_CAM,
        plan_save_path=OUT2_MAP,
        plan_mode="bottom-center",
        cam_trail_len=30,
    )

    # Cam #3
    scst.track_and_save(
        video_path=VIDEO3,
        cctv_pts=cam3_pts,
        plan_pts=cam3_plan_pts,
        plan_img_path=PLAN_IMG_PATH,
        camera_save_path=OUT3_CAM,
        plan_save_path=OUT3_MAP,
        plan_mode="bottom-center",
        cam_trail_len=30,
    )

    print("\n[DONE] All cameras processed.")


if __name__ == "__main__":
    try:
        freeze_support()  # Windows/디버거 spawn 환경 대비
    except Exception:
        pass
    main()
