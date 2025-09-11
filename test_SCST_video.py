import time
from MCMT_engine.SCST.video_SCST import videoSCST
from __Detection.detection_api import DetectionAPI
from __Tracking.tracking_api import TrackerAPI



# 1) 도면 이미지 경로 (공통)
plan_path = "/workspace/assets/seocho/Seocho_plan_pts.png"

# 2) 카메라별 도면 기준점 (PLAN)
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
    (1219,1557)  #빨간꼬깔
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

# 3) 카메라별 영상 좌표 (CCTV)
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


class Args:
    track_thresh = 0.3
    match_thresh = 0.9
    track_buffer = 180
    mot20 = False
    cpu_workers = 16   # 최적화: 20 → 16
    chunk_sec = 30.0   # 최적화: 10 → 30 (더 큰 청크)
    batch_size = 20    # 최적화: 20 → 64 (더 큰 배치)

args = Args()


detector = DetectionAPI(
    models=["ultra_best", "worker"],
    thres=0.0,
    device="cuda:0",
    use_async=True,
    max_workers=1,
)
tracker = TrackerAPI(args=args, detector=detector)

# 인스턴스 1개만 생성
scst = videoSCST(
    plan_path=plan_path,
    args=args,
    detector=detector,
    tracker=tracker,
    ransac_thresh=3.0,
)

# ── Camera 1 ──
start = time.time()
res1 = scst.track_and_save(
    video_path="/workspace/datasets/250909_Site_seocho/2025-09-09 13_29_59 이동형 #1.mp4",
    cam_pts=cam1_pts,
    plan_pts=cam1_plan_pts,
    plan_img_path=plan_path,  # projector 생성/갱신
    camera_save_path="/workspace/results/tracking_result1.mp4",
    plan_save_path="/workspace/results/plan_result1.mp4",
    cam_trail_len=30,
    ransac_thresh=3.0,
)
print(f"Camera 1 처리 완료 ({time.time()-start:.2f}초, {len(res1)} 프레임)")

# ── Camera 2 ──
start = time.time()
res2 = scst.track_and_save(
    video_path="/workspace/datasets/250909_Site_seocho/2025-09-09 13_29_59 이동형 #2.mp4",
    cam_pts=cam2_pts,
    plan_pts=cam2_plan_pts,
    plan_img_path=plan_path,  # 동일 도면이므로 계속 전달(안전)
    camera_save_path="/workspace/results/tracking_result2.mp4",
    plan_save_path="/workspace/results/plan_result2.mp4",
    cam_trail_len=30,
    ransac_thresh=3.0,
)
print(f"Camera 2 처리 완료 ({time.time()-start:.2f}초, {len(res2)} 프레임)")

# ── Camera 3 ──
start = time.time()
res3 = scst.track_and_save(
    video_path="/workspace/datasets/250909_Site_seocho/2025-09-09 13_29_59 이동형 #3.mp4",
    cam_pts=cam3_pts,
    plan_pts=cam3_plan_pts,
    plan_img_path=plan_path,  # 동일 도면
    camera_save_path="/workspace/results/tracking_result3.mp4",
    plan_save_path="/workspace/results/plan_result3.mp4",
    cam_trail_len=30,
    ransac_thresh=3.0,
)
print(f"Camera 3 처리 완료 ({time.time()-start:.2f}초, {len(res3)} 프레임)")

# 요약
total_frames = len(res1) + len(res2) + len(res3)
print(f"총 처리 프레임: {total_frames}")
