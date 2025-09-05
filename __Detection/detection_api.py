# /workspace/__Detection/detection_api.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

# 엔진의 EnsembleDetector
from __Detection.ensemble_detection.engine.base import EnsembleDetector


class DetectionAPI:
    """
    DetectionAPI: 여러 등록된 Detector를 관리하는 고수준 인터페이스.
    - detect()       : torch.Tensor (N,6) 반환 [x1,y1,x2,y2,score,class_id]
    - detect_batch(): List[torch.Tensor] 반환 (프레임 리스트 입력용)
    - detect_raw()   : detector별 원시 dict 반환
    - name_map()     : class_id → label 맵
    """

    def __init__(
        self,
        thres: float = 0.0,
        device: str = "cuda",
        models: Optional[List[str]] = None,        # ["vehicle","worker"]
        exclude: Optional[List[str]] = None,       # ["deprecated_model"]
        device_map: Optional[Dict[str, str]] = None,  # {"vehicle":"cuda:0","worker":"cuda:1"}
        class_names: Optional[List[str]] = None,   # ["Person","Bicycle",...], None이면 coco2name 사용
        use_async: bool = True,
        max_workers: Optional[int] = None,
    ):
        self.device = device
        self.detector = EnsembleDetector(
            thres=thres,
            names=models,
            exclude=exclude,
            device_map=device_map,
            use_async=use_async,
            max_workers=max_workers,
        )
        self._custom_class_map: Dict[int, str] = self._init_class_map(class_names)

    # ----------------------------
    # 내부: 클래스 맵 초기화
    # ----------------------------
    def _init_class_map(self, class_names: Optional[List[str]]) -> Dict[int, str]:
        """사용자 정의 class_names가 있으면 인덱스 기반 맵 생성"""
        if class_names:
            return {i: name for i, name in enumerate(class_names)}
        return {}

    # ----------------------------
    # 내부: list[dict] → Tensor(N,6) 표준화
    # ----------------------------
    def _results_to_tensor(self, results: List[Dict[str, Any]]) -> torch.Tensor:
        if not results:
            return torch.zeros((0, 6), dtype=torch.float32, device=self.device)

        dets: List[List[float]] = []
        append = dets.append  # 미세 최적화
        for r in results:
            # 필수 키 검사
            bbox = r.get("bbox", None)
            score = r.get("score", None)
            cid = r.get("class_id", None)
            if bbox is None or score is None or cid is None:
                continue
            x1, y1, x2, y2 = bbox
            append([float(x1), float(y1), float(x2), float(y2), float(score), float(cid)])

        if not dets:
            return torch.zeros((0, 6), dtype=torch.float32, device=self.device)

        # target device로 바로 생성 (불필요한 .to() 방지)
        return torch.tensor(dets, dtype=torch.float32, device=self.device)

    # ----------------------------
    # 컨텍스트/리소스
    # ----------------------------
    def close(self):
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ----------------------------
    # 유틸: 이미지 크기
    # ----------------------------
    def imgInfo(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Args:
            image: numpy.ndarray (HxWxC, BGR)
        Returns:
            (height, width)
        """
        if image is None:
            raise ValueError("Invalid image (None).")
        return image.shape[:2]

    # ----------------------------
    # 추론: 단일 프레임
    # ----------------------------
    def detect(self, image: np.ndarray) -> torch.Tensor:
        """
        Args:
            image: numpy.ndarray (HxWx3, BGR)
        Returns:
            torch.Tensor, shape (N,6) [x1,y1,x2,y2,score,class_id]
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray (BGR).")

        results = self.detector.detect(image)  # list of dicts(표준 스키마)
        return self._results_to_tensor(results)

    # ----------------------------
    # 추론: 배치 (프레임 리스트)
    # ----------------------------
    def detect_batch(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """
        Args:
            images: List[np.ndarray(BGR)]
        Returns:
            List[Tensor (N,6) on self.device]  # 프레임별 한 개의 텐서
        Note:
            - EnsembleDetector.detect_batch(images) 사용
            - 엔진이 배치를 미구현했으면 내부에서 per-frame detect로 폴백됨
        """
        if not isinstance(images, list) or not images:
            return []
        # 엔진에서 프레임별 list[dict]를 프레임 순서대로 반환
        per_frame_results: List[List[Dict[str, Any]]] = self.detector.detect_batch(images)
        # 즉시 Tensor로 표준화해서 메모리 피크를 낮춤
        return [self._results_to_tensor(res) for res in per_frame_results]

    # ----------------------------
    # 추론: 원시 dict (단일 프레임)
    # ----------------------------
    def detect_raw(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns:
            [{"class_id": int, "label": str, "score": float,
              "bbox": [x1,y1,x2,y2], "source": "vehicle"}, ...]
        """
        return self.detector.detect(image)

    # ----------------------------
    # class_id → label 맵
    # ----------------------------
    def name_map(self) -> Dict[int, str]:
        """
        Returns:
            dict: {class_id: label}
        - 사용자가 지정한 class_names가 있으면 우선 적용
        - 아니면 EnsembleDetector 내부 coco2name 사용
        """
        if self._custom_class_map:
            return self._custom_class_map

        merged: Dict[int, str] = {}
        for det in getattr(self.detector, "detectors", []):
            merged.update(getattr(det, "coco2name", {}))
        return merged
