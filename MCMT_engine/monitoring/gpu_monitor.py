import time
import logging
from typing import Optional, Tuple
import subprocess
import re

class GPUMonitor:
    """GPU 사용률 모니터링 클래스"""
    
    def __init__(self, gpu_id: int = 0, threshold: float = 95.0):
        """
        Args:
            gpu_id: 모니터링할 GPU ID (기본값: 0)
            threshold: GPU 사용률 임계값 (기본값: 95%)
        """
        self.gpu_id = gpu_id
        self.threshold = threshold
        self.last_check_time = 0
        self.check_interval = 0.5  # 0.5초마다 체크
        self.last_utilization = 0.0
        
    def get_gpu_utilization(self) -> float:
        """현재 GPU 사용률을 반환합니다."""
        try:
            # nvidia-smi를 사용하여 GPU 사용률 조회
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu', 
                '--format=csv,noheader,nounits',
                f'--id={self.gpu_id}'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                utilization = float(result.stdout.strip())
                self.last_utilization = utilization
                return utilization
            else:
                logging.warning(f"nvidia-smi failed: {result.stderr}")
                return self.last_utilization
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
            logging.warning(f"GPU monitoring failed: {e}")
            return self.last_utilization
    
    def is_gpu_overloaded(self) -> bool:
        """GPU가 과부하 상태인지 확인합니다."""
        current_time = time.time()
        
        # 체크 간격을 두어 성능 최적화
        if current_time - self.last_check_time < self.check_interval:
            return self.last_utilization >= self.threshold
            
        self.last_check_time = current_time
        utilization = self.get_gpu_utilization()
        
        return utilization >= self.threshold
    
    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """GPU 메모리 사용량을 반환합니다 (사용량, 전체용량)."""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.used,memory.total', 
                '--format=csv,noheader,nounits',
                f'--id={self.gpu_id}'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                used, total = map(float, result.stdout.strip().split(', '))
                return used, total
            else:
                return 0.0, 0.0
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError):
            return 0.0, 0.0
    
    def log_gpu_status(self):
        """GPU 상태를 로깅합니다."""
        utilization = self.get_gpu_utilization()
        used_mem, total_mem = self.get_gpu_memory_usage()
        mem_percent = (used_mem / total_mem * 100) if total_mem > 0 else 0
        
        logging.info(f"GPU{self.gpu_id}: {utilization:.1f}% util, {mem_percent:.1f}% mem ({used_mem:.0f}MB/{total_mem:.0f}MB)")
