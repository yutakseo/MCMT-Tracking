import time
import logging
import psutil
import torch
from typing import Dict, Any

class HealthMonitor:
    """시스템 건강 상태 모니터링"""
    
    def __init__(self):
        self.last_check_time = 0
        self.check_interval = 1.0
        self.stall_threshold = 10.0  # 10초 이상 응답 없으면 스톨로 판단
        
    def check_system_health(self) -> Dict[str, Any]:
        """시스템 건강 상태 체크"""
        current_time = time.time()
        
        # 체크 간격 제한
        if current_time - self.last_check_time < self.check_interval:
            return self._get_cached_health()
            
        self.last_check_time = current_time
        
        health = {
            'timestamp': current_time,
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'gpu_memory_used': 0,
            'gpu_memory_total': 0,
            'gpu_utilization': 0,
            'is_healthy': True,
            'warnings': []
        }
        
        # GPU 메모리 체크
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                health['gpu_memory_used'] = gpu_memory
                health['gpu_memory_total'] = gpu_memory_total
                health['gpu_memory_percent'] = (gpu_memory / gpu_memory_total) * 100
                
                # GPU 메모리 90% 이상 시 경고
                if health['gpu_memory_percent'] > 90:
                    health['warnings'].append(f"GPU memory high: {health['gpu_memory_percent']:.1f}%")
                    health['is_healthy'] = False
        except Exception as e:
            health['warnings'].append(f"GPU check failed: {e}")
        
        # CPU 95% 이상 시 경고
        if health['cpu_percent'] > 95:
            health['warnings'].append(f"CPU high: {health['cpu_percent']:.1f}%")
            health['is_healthy'] = False
            
        # 메모리 90% 이상 시 경고
        if health['memory_percent'] > 90:
            health['warnings'].append(f"Memory high: {health['memory_percent']:.1f}%")
            health['is_healthy'] = False
            
        return health
    
    def _get_cached_health(self) -> Dict[str, Any]:
        """캐시된 건강 상태 반환"""
        return {
            'timestamp': time.time(),
            'is_healthy': True,
            'warnings': []
        }
    
    def log_health_status(self):
        """건강 상태 로깅"""
        health = self.check_system_health()
        if health['warnings']:
            logging.warning(f"Health warnings: {', '.join(health['warnings'])}")
        else:
            logging.info(f"System healthy - CPU: {health.get('cpu_percent', 0):.1f}%, "
                        f"Memory: {health.get('memory_percent', 0):.1f}%, "
                        f"GPU: {health.get('gpu_memory_percent', 0):.1f}%")
    
    def should_skip_processing(self) -> bool:
        """처리 스킵 여부 판단"""
        health = self.check_system_health()
        return not health['is_healthy']
