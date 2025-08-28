# /workspace/__Detection/detection_api.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

# 엔진의 EnsembleDetector
from __Detection.ensemble_detection.engine.base import EnsembleDetector

class DetectionAPI:
    def __init__(
        self,
        thres: float = 0.0,
        device: str = "cuda",
        names: Optional[List[str]] = None,                # ["vehicle","worker"]
        exclude: Optional[List[str]] = None,              # ["deprecated_model"]
        device_map: Optional[Dict[str, str]] = None,      # {"vehicle":"cuda:0","worker":"cuda:1"}
        use_async: bool = True,
        max_workers: Optional[int] = None,
    ):
        self.detector = EnsembleDetector(
            thres=thres,
            names=names,
            exclude=exclude,
            device_map=device_map,
            use_async=use_async,
            max_workers=max_workers,
        )
        self.device = device  # 반환 텐서가 올라갈 디바이스 ("cuda" / "cpu")

    # 리소스 정리용 (선택)
    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # 간단 정보 출력
    def imgInfo(self, image:np.ndarray) -> Tuple[int, int]:
        """
        image: numpy.ndarray (HxWxC, BGR)
        return: (height, width)
        """
        if image is None:
            raise ValueError("Invalid image (None).")
        h, w = image.shape[:2]
        return (h, w)

    # 표준 출력: (N, 6) [x1,y1,x2,y2,score,class_id]
    def detect(self, image:np.ndarray) -> torch.Tensor:
        """
        image: numpy.ndarray (HxWx3, BGR)
        return: torch.Tensor, shape (N,6), dtype float32 on self.device
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray (BGR).")

        results = self.detector.detect(image)  # list of dicts( bbox/score/class_id/... )

        if not results:
            return torch.zeros((0, 6), dtype=torch.float32, device=self.device)

        dets: List[List[float]] = []
        for r in results:
            try:
                x1, y1, x2, y2 = r["bbox"]
                score = float(r["score"])
                cid   = int(r["class_id"])
                dets.append([float(x1), float(y1), float(x2), float(y2), score, float(cid)])
            except Exception:
                # 필수 키가 없으면 스킵
                continue

        if not dets:
            return torch.zeros((0, 6), dtype=torch.float32, device=self.device)

        out = torch.tensor(dets, dtype=torch.float32)
        return out.to(self.device, non_blocking=True)

    # 원시(dict) 결과가 필요할 때
    def detect_raw(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        반환 예시:
        {"class_id": int, "label": str, "score": float, "bbox": [x1,y1,x2,y2], "source": "vehicle" }
        """
        return self.detector.detect(image)

    # label 조회용 헬퍼 (시각화에 유용)
    def name_map(self) -> Dict[int, str]:
        """
        모든 등록 디텍터의 coco2name을 합쳐 class_id -> name 맵을 생성
        """
        m: Dict[int, str] = {}
        for det in getattr(self.detector, "detectors", []):
            m.update(getattr(det, "coco2name", {}))
        return m
