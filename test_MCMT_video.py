# /workspace/test_MCMT_video.py - 멀티카메라 비디오 처리 테스트
"""
MCMT_video.py 테스트 스크립트
"""

import os
import sys
from pathlib import Path

sys.path.append("/workspace")

from MCMT_video import create_video_tracking_system

def test_multi_camera_video_processing():
    """멀티카메라 비디오 처리 테스트"""
    
    # 테스트용 비디오 파일 경로 (실제 파일이 있는지 확인)
    video_paths = [
        "/workspace/datasets/homography_experiment2/1100-1110/2025-09-04 10_59_59 이동형 #1_part_000.mp4",
        "/workspace/datasets/homography_experiment2/1100-1110/2025-09-04 10_59_58 이동형 #2_part_000.mp4",
        "/workspace/datasets/homography_experiment2/1100-1110/2025-09-04 10_59_58 이동형 #3_part_000.mp4",
    ]
    
    # 파일 존재 확인
    existing_videos = []
    for video_path in video_paths:
        if os.path.exists(video_path):
            existing_videos.append(video_path)
            print(f"✅ 비디오 파일 확인: {video_path}")
        else:
            print(f"❌ 비디오 파일 없음: {video_path}")
    
    if not existing_videos:
        print("❌ 처리할 비디오 파일이 없습니다.")
        return False
    
    print(f"\n📹 {len(existing_videos)}개 비디오 파일로 테스트 시작")
    
    # 멀티카메라 비디오 시스템 생성
    system = create_video_tracking_system(
        video_paths=existing_videos,
        detector_models=["ultra_people", "worker"],
        class_names=["People"],
        output_dir="/workspace/results/multi_camera_video_test"
    )
    
    # 시스템 실행
    print("\n🚀 멀티카메라 비디오 처리 시작...")
    success = system.run(
        parallel=True,  # 병렬 처리
        max_workers=min(3, len(existing_videos)),  # 최대 워커 수
        create_unified=True  # 통합 플랜 비디오 생성
    )
    
    if success:
        print("\n✅ 멀티카메라 비디오 처리 완료!")
        print(f"📁 결과 저장 위치: {system.output_dir}")
        
        # 결과 파일 확인
        result_files = list(system.output_dir.glob("*"))
        print(f"\n📄 생성된 파일들:")
        for file in result_files:
            print(f"  - {file.name}")
        
        return True
    else:
        print("\n❌ 멀티카메라 비디오 처리 실패!")
        return False

if __name__ == "__main__":
    print("=== 멀티카메라 비디오 처리 테스트 ===")
    test_multi_camera_video_processing()
