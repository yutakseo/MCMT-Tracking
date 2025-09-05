# /workspace/MCMT_engine/async_inference.py
from __future__ import annotations

import asyncio
import time
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from MCMT_engine.stream_SCST import streamSCST


class AsyncEngine:
    """
    멀티 카메라 동기화 추론 엔진 (단순화 버전)
    
    사용법:
        engine = AsyncEngine(cameras)
        async for result in engine.stream():
            print(f"Time: {result['timestamp']}")
            print(f"Cameras: {len(result['cameras'])}")
    """
    
    def __init__(self, cameras: List[streamSCST], interval: float = 0.1):
        self.cameras = cameras
        self.interval = interval
        self.is_running = False
        
    async def stream(self):
        """비동기 스트리밍 시작"""
        self.is_running = True
        round_num = 0
        
        try:
            while self.is_running:
                # 1. 모든 카메라에서 동시 캡처
                timestamp = time.time()
                frames = await self._capture_all_cameras()
                
                # 2. 각 카메라별 추론 처리
                camera_results = await self._process_all_cameras(frames, timestamp)
                
                # 3. 결과 패키징
                result = {
                    'round': round_num,
                    'timestamp': timestamp,
                    'cameras': camera_results
                }
                
                # 4. 결과 출력
                self._print_results(result)
                
                # 5. 결과 반환
                yield result
                
                # 6. 다음 라운드까지 대기
                await asyncio.sleep(self.interval)
                round_num += 1
                
        except Exception as e:
            logging.error(f"Streaming error: {e}")
        finally:
            self.is_running = False
    
    async def _capture_all_cameras(self) -> List[Optional[np.ndarray]]:
        """모든 카메라에서 동시 프레임 캡처"""
        tasks = []
        for camera in self.cameras:
            task = asyncio.get_event_loop().run_in_executor(
                None, camera._videoCapture
            )
            tasks.append(task)
        
        frames = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외 처리
        valid_frames = []
        for i, frame in enumerate(frames):
            if isinstance(frame, Exception):
                logging.error(f"Camera {i+1} capture failed: {frame}")
                valid_frames.append(None)
            else:
                valid_frames.append(frame)
        
        return valid_frames
    
    async def _process_all_cameras(
        self, 
        frames: List[Optional[np.ndarray]], 
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """모든 카메라에서 추론 처리"""
        tasks = []
        for i, (camera, frame) in enumerate(zip(self.cameras, frames)):
            if frame is not None:
                task = self._process_single_camera(camera, frame, timestamp, i+1)
                tasks.append(task)
            else:
                # 프레임이 없는 경우 빈 결과
                tasks.append(asyncio.create_task(self._empty_result(i+1)))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 예외 처리
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Camera {i+1} processing failed: {result}")
                valid_results.append(self._empty_result(i+1))
            else:
                valid_results.append(result)
        
        return valid_results
    
    async def _process_single_camera(
        self,
        camera: streamSCST, 
        frame: np.ndarray, 
        timestamp: float, 
        camera_id: int
    ) -> Dict[str, Any]:
        """단일 카메라 처리"""
        try:
            # 1. 객체 탐지
            detection_results = await asyncio.get_event_loop().run_in_executor(
                None, camera._detection, frame
            )
            
            # 2. 추적
            tracklets = await asyncio.get_event_loop().run_in_executor(
                None, camera._tracking, frame
            )
            
            # 3. 호모그래피 변환
            projected, _ = await asyncio.get_event_loop().run_in_executor(
                None, camera._projection, tracklets, timestamp
            )
            
            # 4. 도면 좌표 추출
            plan_coords = [item["pt"] for item in projected]
            
            return {
                'camera_id': camera_id,
                'detection_count': len(detection_results),
                'tracking_count': len(tracklets),
                'plan_coords': plan_coords,
                'tracklets': tracklets,
                'projected': projected
            }
            
        except Exception as e:
            logging.error(f"Camera {camera_id} processing error: {e}")
            return self._empty_result(camera_id)
    
    async def _empty_result(self, camera_id: int) -> Dict[str, Any]:
        """빈 결과 반환"""
        return {
            'camera_id': camera_id,
            'detection_count': 0,
            'tracking_count': 0,
            'plan_coords': [],
            'tracklets': [],
            'projected': []
        }
    
    def _print_results(self, result: Dict[str, Any]):
        """결과 출력"""
        timestamp = result['timestamp']
        round_num = result['round']
        
        print(f"\n=== Round {round_num} at {timestamp:.3f} ===")
        
        for cam_result in result['cameras']:
            cam_id = cam_result['camera_id']
            det_count = cam_result['detection_count']
            track_count = cam_result['tracking_count']
            coords = cam_result['plan_coords']
            
            print(f"Camera {cam_id}: {det_count} detections, {track_count} tracks")
            for i, (x, y) in enumerate(coords):
                print(f"  Object {i+1}: ({x:.1f}, {y:.1f})")
    
    def stop(self):
        """스트리밍 중지"""
        self.is_running = False


# 사용 예시
async def main():
    # 카메라 설정 (streamSCST 인스턴스들)
    cameras = [
        streamSCST(
            cctv_url="rtsp://192.168.1.100:554/stream1",
            cctv_benchmark=[(100, 200), (300, 400), (500, 600), (700, 800)],
            plan_path="floor_plan.jpg",
            plan_benchmark=[(50, 100), (150, 200), (250, 300), (350, 400)],
            tracker_args=None
        ),
        streamSCST(
            cctv_url="rtsp://192.168.1.101:554/stream1",
            cctv_benchmark=[(120, 220), (320, 420), (520, 620), (720, 820)],
            plan_path="floor_plan.jpg",
            plan_benchmark=[(60, 110), (160, 210), (260, 310), (360, 410)],
            tracker_args=None
        ),
        streamSCST(
            cctv_url="rtsp://192.168.1.102:554/stream1",
            cctv_benchmark=[(140, 240), (340, 440), (540, 640), (740, 840)],
            plan_path="floor_plan.jpg",
            plan_benchmark=[(70, 120), (170, 220), (270, 320), (370, 420)],
            tracker_args=None
        )
    ]
    
    # 엔진 생성
    engine = AsyncEngine(cameras, interval=0.1)
    
    try:
        # 스트리밍 시작
        async for result in engine.stream():
            # 여기서 결과 처리
            timestamp = result['timestamp']
            camera_data = result['cameras']
            
            # 예: 도면 좌표만 추출
            all_coords = []
            for cam_data in camera_data:
                all_coords.extend(cam_data['plan_coords'])
            
            print(f"Total objects: {len(all_coords)}")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
