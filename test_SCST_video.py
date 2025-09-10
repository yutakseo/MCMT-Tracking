# === Import (새 클래스 우선, 실패 시 폴백) ===
import time
import torch
from MCMT_engine.SCST.video_SCST import videoSCST
from __Detection.detection_api import DetectionAPI
from __Tracking.tracking_api import TrackerAPI
from MCMT_engine.core.homoGraphy import PlanProjector

# === 최적화된 Args ===
class Args:
    track_thresh = 0.3
    match_thresh = 0.9
    track_buffer = 180
    mot20 = False
    cpu_workers = 16   # 최적화: 20 → 16
    chunk_sec = 30.0   # 최적화: 10 → 30 (더 큰 청크)
    batch_size = 64    # 최적화: 20 → 64 (더 큰 배치)

args = Args()

# 1) 도면 이미지 경로 (공통)
plan_path = "/workspace/assets/seocho/Seocho_plan_pts.png"

# 2) 카메라별 도면 기준점 (PLAN)  ── 주석/줄바꿈 유지
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

# 3) 카메라별 영상 좌표 (CCTV)  ── 주석/줄바꿈 유지
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

def main():
    start_time = time.time()
    print("🚀 멀티카메라 비디오 처리 시작 (최적화 버전)")
    
    # GPU 메모리 최적화
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        print(f"✅ GPU 최적화 완료: {torch.cuda.get_device_name()}")
    
    # =============================================================================
    # 1. 공유 모델 인스턴스 생성 (한 번만!)
    # =============================================================================
    print("📦 공유 모델 초기화 중...")
    model_init_start = time.time()
    
    # 공유 객체탐지 모델
    shared_detector = DetectionAPI(
        models=["ultra_people", "worker"],
        thres=0.0,
        device="cuda:0",
        use_async=True,
        max_workers=1
    )
    
    # 공유 추적기 모델
    shared_tracker = TrackerAPI(args=args, detector=shared_detector)
    
    # 공유 호모그래피 인스턴스 (임시로 첫 번째 카메라 포인트 사용)
    shared_projector = PlanProjector(
        plan_img_or_path=plan_path,
        image_pts=cam1_pts,
        plan_pts=cam1_plan_pts,
    )
    
    model_init_time = time.time() - model_init_start
    print(f"✅ 공유 모델 초기화 완료 ({model_init_time:.2f}초)")
    
    # =============================================================================
    # 2. 호모그래피 미리 계산 (캐싱)
    # =============================================================================
    print("🎯 호모그래피 캐싱 중...")
    homography_cache = {}
    
    camera_configs = [
        (cam1_pts, cam1_plan_pts, "Camera 1"),
        (cam2_pts, cam2_plan_pts, "Camera 2"), 
        (cam3_pts, cam3_plan_pts, "Camera 3")
    ]
    
    for i, (cctv_pts, plan_pts, cam_name) in enumerate(camera_configs):
        H, _ = shared_projector.fit_homography(cctv_pts, plan_pts)
        homography_cache[i] = H
        print(f"✅ {cam_name} 호모그래피 계산 완료")
    
    # =============================================================================
    # 3. 각 카메라별 처리 (공유 모델 사용)
    # =============================================================================
    video_paths = [
        "/workspace/datasets/250909_Site_seocho/2025-09-09 13_29_59 이동형 #1.mp4",
        "/workspace/datasets/250909_Site_seocho/2025-09-09 13_29_59 이동형 #2.mp4",
        "/workspace/datasets/250909_Site_seocho/2025-09-09 13_29_59 이동형 #3.mp4",
    ]
    
    results = []
    
    for i, (video_path, (cctv_pts, plan_pts, cam_name)) in enumerate(zip(video_paths, camera_configs)):
        print(f"\n📹 {cam_name} 처리 시작...")
        cam_start = time.time()
        
        # videoSCST 인스턴스 생성 (공유 모델 주입)
        scst = videoSCST(
            args=args,
            plan_img_path=plan_path,
            plan_pts=plan_pts,
            det_models=["ultra_people", "worker"]
        )
        
        # 공유 모델들로 교체
        scst.detector = shared_detector
        scst.tracker = shared_tracker
        scst.projector = shared_projector
        
        # 캐시된 호모그래피 설정
        scst.H = homography_cache[i]
        scst._last_cctv_pts = list(cctv_pts)
        scst._last_plan_pts = list(plan_pts)
        scst._last_H = homography_cache[i]
        
        # 비디오 처리
        result = scst.track_and_save(
            video_path=video_path,
            cam_pts=cctv_pts,
            plan_pts=plan_pts,
            plan_img_path=plan_path,
            camera_save_path=f"/workspace/results/tracking_result{i+1}.mp4",
            plan_save_path=f"/workspace/results/plan_result{i+1}.mp4",
        )
        
        results.append(result)
        cam_time = time.time() - cam_start
        print(f"✅ {cam_name} 처리 완료 ({cam_time:.2f}초, {len(result)} 프레임)")
    
    # =============================================================================
    # 4. 결과 요약
    # =============================================================================
    total_time = time.time() - start_time
    total_frames = sum(len(r) for r in results)
    
    print(f"\n🎉 모든 카메라 처리 완료!")
    print(f"⏱️  총 처리 시간: {total_time:.2f}초")
    print(f"📊 총 처리 프레임: {total_frames:,}개")
    print(f"🚀 평균 처리 속도: {total_frames/total_time:.1f} FPS")
    print(f"💾 결과 저장 위치: /workspace/results/")
    
    # 리소스 정리
    try:
        shared_detector.close()
        print("🧹 리소스 정리 완료")
    except Exception as e:
        print(f"⚠️  리소스 정리 중 오류: {e}")




if __name__ == "__main__":
    # Windows/Spawn 대비(얼어붙인 실행파일 아닐 때는 없어도 무방)
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass

    main()