# /workspace/MCMT_engine/async_inference.py - 스마트 GPU 관리 비동기 추론 엔진

from __future__ import annotations

import asyncio
import time
import logging
import torch
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from ..streaming.stream_SCST import streamSCST
from ..monitoring.gpu_monitor import GPUMonitor


class AsyncEngine:
    """
    스마트 GPU 관리 멀티 카메라 엔진
    - GPU 90% 이상: 프레임 스킵 (성능 보호)
    - GPU 95% 이상: 추론 스킵 (안정성 보호)  
    - GPU 98% 이상: 완전 정지 (시스템 보호)
    """
    
    def __init__(
        self,
        cameras: List[streamSCST], 
        interval: float = 0.1,
        gpu_warning: float = 90.0,    # 90% 이상: 프레임 스킵
        gpu_danger: float = 95.0,     # 95% 이상: 추론 스킵
        gpu_critical: float = 98.0,   # 98% 이상: 완전 정지
        gpu_recovery: float = 85.0,   # 85% 이하: 정상 복구
        gpu_id: int = 0
    ):
        self.cameras = cameras
        self.interval = interval
        self.is_running = False
        
        # GPU 모니터링 (다단계 임계값)
        self.gpu_monitor = GPUMonitor(gpu_id=gpu_id, threshold=gpu_critical)
        self.gpu_warning_threshold = gpu_warning
        self.gpu_danger_threshold = gpu_danger
        self.gpu_critical_threshold = gpu_critical
        self.gpu_recovery_threshold = gpu_recovery
        
        # 상태 관리
        self.gpu_state = "NORMAL"  # NORMAL, WARNING, DANGER, CRITICAL
        self.state_start_time = 0
        
        # 통계
        self.stats = {
            'total_rounds': 0,
            'normal_rounds': 0,
            'warning_rounds': 0,
            'danger_rounds': 0,
            'critical_rounds': 0,
            'total_detections': 0,
            'total_tracks': 0,
            'batch_processing_time': 0.0
        }
        
        logging.info(f"✅ AsyncEngine 초기화 완료 - {len(cameras)}개 카메라, 스마트 GPU 관리")
        
    async def stream(self):
        """스마트 GPU 관리 스트리밍"""
        self.is_running = True
        round_num = 0
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # GPU 상태 체크
                current_gpu_util = self.gpu_monitor.get_gpu_utilization()
                new_state = self._determine_gpu_state(current_gpu_util)
                
                # 상태 변화 감지
                if new_state != self.gpu_state:
                    old_state = self.gpu_state
                    self.gpu_state = new_state
                    self.state_start_time = current_time
                    self._log_state_change(old_state, new_state, current_gpu_util)
                
                # GPU 상태에 따른 처리
                if self.gpu_state == "CRITICAL":
                    await self._handle_critical_state()
                    self.stats['critical_rounds'] += 1
                    
                elif self.gpu_state == "DANGER":
                    await self._handle_danger_state()
                    self.stats['danger_rounds'] += 1
                    
                elif self.gpu_state == "WARNING":
                    await self._handle_warning_state()
                    self.stats['warning_rounds'] += 1
                    
                else:
                    # 정상 처리 - 배치 처리로 모든 카메라 동시 처리
                    result = await self._process_batch_round(round_num, current_gpu_util)
                    self.stats['normal_rounds'] += 1
                    if result:
                        yield result
                
                # 통계 업데이트
                self.stats['total_rounds'] += 1
                
                # 다음 라운드까지 대기
                await asyncio.sleep(self.interval)
                round_num += 1
                
        except Exception as e:
            logging.error(f"Streaming error: {e}")
        finally:
            self.is_running = False
            self._log_final_stats()

    async def _process_batch_round(self, round_num: int, gpu_util: float):
        """배치 처리로 모든 카메라 동시 처리"""
        try:
            timestamp = time.time()
            print(f"[DEBUG] AsyncEngine._process_batch_round: round={round_num}, gpu_util={gpu_util:.1f}%")
            
            # 1. 모든 카메라에서 프레임 캡처
            print(f"[DEBUG] AsyncEngine: 프레임 캡처 시작...")
            frames = await self._capture_all_cameras()
            print(f"[DEBUG] AsyncEngine: 프레임 캡처 완료, {len(frames)}개 프레임")
            
            # 2. 유효한 프레임들만 필터링
            valid_frames = [(i, frame) for i, frame in enumerate(frames) 
                          if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0]
            print(f"[DEBUG] AsyncEngine: 유효한 프레임 {len(valid_frames)}개")
            
            if not valid_frames:
                print(f"[DEBUG] AsyncEngine: 유효한 프레임이 없음, None 반환")
                return None
            
            # 3. 배치 탐지 (단일 모델로 모든 프레임 처리)
            print(f"[DEBUG] AsyncEngine: 배치 탐지 시작...")
            batch_start_time = time.time()
            detection_results = await self._batch_detection([frame for _, frame in valid_frames])
            batch_time = time.time() - batch_start_time
            print(f"[DEBUG] AsyncEngine: 배치 탐지 완료, {len(detection_results)}개 결과, {batch_time:.3f}s")
            
            # 4. 각 카메라별 추적 및 호모그래피 처리
            camera_results = []
            for i, (cam_idx, frame) in enumerate(valid_frames):
                det_result = detection_results[i] if i < len(detection_results) else []
                
                # detection 결과를 numpy로 변환
                det_result = self._convert_to_numpy(det_result)
                
                print(f"[DEBUG] AsyncEngine: Camera {cam_idx+1} - {len(det_result)} detections")
                
                # 추적
                print(f"[DEBUG] AsyncEngine: Camera {cam_idx+1} 추적 시작...")
                tracklets = await asyncio.get_event_loop().run_in_executor(
                    None, self.cameras[cam_idx]._tracking, frame
                )
                print(f"[DEBUG] AsyncEngine: Camera {cam_idx+1} 추적 완료, {len(tracklets)} tracklets")
                
                # 호모그래피 변환
                print(f"[DEBUG] AsyncEngine: Camera {cam_idx+1} 호모그래피 변환 시작...")
                projected, _ = await asyncio.get_event_loop().run_in_executor(
                    None, self.cameras[cam_idx]._projection, tracklets
                )
                print(f"[DEBUG] AsyncEngine: Camera {cam_idx+1} 호모그래피 변환 완료, {len(projected)} projected")
                
                # 도면 좌표 추출
                plan_coords = [item.get("pt") for item in projected 
                              if isinstance(item, dict) and "pt" in item]
                print(f"[DEBUG] AsyncEngine: Camera {cam_idx+1} plan_coords: {len(plan_coords)}개")
                
                h, w = frame.shape[:2]
                camera_result = {
                    'camera_id': cam_idx + 1,
                    'detection_count': len(det_result) if hasattr(det_result, "__len__") else 0,
                    'tracking_count': len(tracklets) if hasattr(tracklets, "__len__") else 0,
                    'plan_coords': plan_coords,
                    'tracklets': tracklets,
                    'projected': projected,
                    'frame_shape': (h, w),
                    'detections': det_result,
                }
                camera_results.append(camera_result)
                print(f"[DEBUG] AsyncEngine: Camera {cam_idx+1} 결과: {camera_result}")
                
                # 통계 업데이트
                self.stats['total_detections'] += len(det_result) if hasattr(det_result, "__len__") else 0
                self.stats['total_tracks'] += len(tracklets) if hasattr(tracklets, "__len__") else 0
            
            # 빈 카메라 결과 추가
            for i in range(len(self.cameras)):
                if not any(result['camera_id'] == i + 1 for result in camera_results):
                    camera_results.append(self._empty_result(i + 1))
            
            self.stats['batch_processing_time'] = batch_time
            
            result = {
                'round': round_num,
                'timestamp': timestamp,
                'cameras': camera_results,
                'gpu_state': self.gpu_state,
                'gpu_utilization': gpu_util,
                'batch_time': batch_time,
                'total_detections': sum(r['detection_count'] for r in camera_results),
                'total_tracks': sum(r['tracking_count'] for r in camera_results)
            }
            
            self._print_results(result)
            return result
            
        except Exception as e:
            logging.error(f"배치 처리 실패: {e}")
            return None

    async def _batch_detection(self, frames: List[np.ndarray]) -> List[Any]:
        """배치 탐지 - 단일 모델로 여러 프레임 동시 처리"""
        try:
            # 첫 번째 카메라의 공유 모델 사용
            if not self.cameras or not hasattr(self.cameras[0], 'detector') or self.cameras[0].detector is None:
                return [[] for _ in frames]
            
            detector = self.cameras[0].detector
            
            # DetectionAPI의 배치 처리 기능 활용
            if hasattr(detector, 'detect_batch'):
                # 배치 처리 지원하는 경우
                results = await asyncio.get_event_loop().run_in_executor(
                    None, detector.detect_batch, frames
                )
                return results
            else:
                # 개별 처리 (병렬)
                tasks = []
                for frame in frames:
                    task = asyncio.get_event_loop().run_in_executor(
                        None, detector.detect, frame
                    )
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 예외 처리
                valid_results = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logging.error(f"Frame {i} detection failed: {result}")
                        valid_results.append([])
                    else:
                        valid_results.append(result)
                
                return valid_results
                
        except Exception as e:
            logging.error(f"배치 탐지 실패: {e}")
            return [[] for _ in frames]

    async def _capture_all_cameras(self) -> List[Optional[np.ndarray]]:
        """모든 카메라에서 동시 프레임 캡처"""
        tasks = []
        for camera in self.cameras:
            task = asyncio.get_event_loop().run_in_executor(
                None, camera._videoCapture
            )
            tasks.append(task)
        
        frames = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_frames = []
        for i, frame in enumerate(frames):
            if isinstance(frame, Exception):
                logging.error(f"Camera {i+1} capture failed: {frame}")
                valid_frames.append(None)
            else:
                valid_frames.append(frame)
        
        return valid_frames

    def _determine_gpu_state(self, gpu_util: float) -> str:
        """GPU 사용률에 따른 상태 결정"""
        if gpu_util >= self.gpu_critical_threshold:
            return "CRITICAL"
        elif gpu_util >= self.gpu_danger_threshold:
            return "DANGER"
        elif gpu_util >= self.gpu_warning_threshold:
            return "WARNING"
        else:
            return "NORMAL"

    def _log_state_change(self, old_state: str, new_state: str, gpu_util: float):
        """상태 변화 로깅"""
        if new_state == "CRITICAL":
            logging.error(f"🚨 GPU {gpu_util:.1f}% → CRITICAL (완전 정지)")
        elif new_state == "DANGER":
            logging.warning(f"⚠️ GPU {gpu_util:.1f}% → DANGER (추론 스킵)")
        elif new_state == "WARNING":
            logging.warning(f"⚠️ GPU {gpu_util:.1f}% → WARNING (프레임 스킵)")
        elif new_state == "NORMAL":
            logging.info(f"✅ GPU {gpu_util:.1f}% → NORMAL (정상 처리)")

    async def _handle_critical_state(self):
        """CRITICAL 상태 처리: 완전 정지"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        await asyncio.sleep(2.0)

    async def _handle_danger_state(self):
        """DANGER 상태 처리: 추론만 스킵"""
        await asyncio.sleep(0.5)

    async def _handle_warning_state(self):
        """WARNING 상태 처리: 프레임 스킵"""
        await asyncio.sleep(0.3)

    def _empty_result(self, camera_id: int) -> Dict[str, Any]:
        """빈 결과 반환"""
        return {
            'camera_id': camera_id,
            'detection_count': 0,
            'tracking_count': 0,
            'plan_coords': [],
            'tracklets': [],
            'projected': [],
            'frame_shape': None,
        }

    def _print_results(self, result: Dict[str, Any]):
        """결과 출력"""
        timestamp = result['timestamp']
        round_num = result['round']
        gpu_state = result.get('gpu_state', 'UNKNOWN')
        gpu_util = result.get('gpu_utilization', 0)
        batch_time = result.get('batch_time', 0)
        total_detections = result.get('total_detections', 0)
        total_tracks = result.get('total_tracks', 0)
        
        print(f"\n=== Round {round_num} at {timestamp:.3f} [{gpu_state}] GPU:{gpu_util:.1f}% ===")
        print(f"📊 Batch Time: {batch_time:.3f}s, Detections: {total_detections}, Tracks: {total_tracks}")
        
        if gpu_state == "NORMAL":
            for cam_result in result['cameras']:
                cam_id = cam_result['camera_id']
                det_count = cam_result['detection_count']
                track_count = cam_result['tracking_count']
                coords = cam_result['plan_coords']
                
                print(f"Camera {cam_id}: {det_count} detections, {track_count} tracks")
                for i, pt in enumerate(coords):
                    try:
                        x, y = pt
                        print(f"  Object {i+1}: ({x:.1f}, {y:.1f})")
                    except Exception:
                        pass
        else:
            print(f"🛑 GPU {gpu_state} 상태 - 처리 제한")

    def _log_final_stats(self):
        """최종 통계 로깅"""
        stats = self.stats
        total = stats['total_rounds']
        normal = stats['normal_rounds']
        warning = stats['warning_rounds']
        danger = stats['danger_rounds']
        critical = stats['critical_rounds']
        
        logging.info(f"Final Stats - Total: {total}")
        logging.info(f"Normal: {normal}, Warning: {warning}, Danger: {danger}, Critical: {critical}")
        logging.info(f"Total Detections: {stats['total_detections']}, Total Tracks: {stats['total_tracks']}")
        logging.info(f"Avg Batch Time: {stats['batch_processing_time']:.3f}s")
        
        if total > 0:
            logging.info(f"Normal Rate: {normal/total*100:.1f}%, Warning Rate: {warning/total*100:.1f}%")
            logging.info(f"Danger Rate: {danger/total*100:.1f}%, Critical Rate: {critical/total*100:.1f}%")

    def _convert_to_numpy(self, data):
        """tensor를 numpy로 변환하는 유틸리티 함수"""
        if data is None:
            return []
        
        # tensor인 경우
        if hasattr(data, 'cpu'):
            return data.cpu().numpy()
        
        # 이미 numpy array인 경우
        if isinstance(data, np.ndarray):
            return data
        
        # 리스트나 다른 iterable인 경우
        if hasattr(data, '__iter__'):
            try:
                return np.array(data)
            except:
                return data
        
        return data

    def stop(self):
        """스트리밍 중지"""
        self.is_running = False
