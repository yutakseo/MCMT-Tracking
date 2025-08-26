#/workspace/__Detection/detection_api.py
from __Detection.ensemble_detection._base_ import EnsembleDetector
import numpy as np
import torch

class DetectionAPI:
    def __init__(self, thres: float = 0.0, device="cuda"):
        self.detector = EnsembleDetector(thres=thres)
        self.device = device  # GPU에서 바로 쓰고 싶으면 "cuda"

    def imgInfo(self, image):
        """
        image: numpy.ndarray (HxWx3, BGR)
        return: (height, width)
        """
        if image is None:
            raise ValueError("Invalid image (None).")
        h, w = image.shape[:2]
        return (h, w)

    def detect(self, image):
        """
        image: numpy.ndarray (HxWx3, BGR)
        return: torch.Tensor of shape (N, 6)
                format: [x1, y1, x2, y2, score, class_id]
        """
        results = self.detector.detect(image)  # list of dicts

        dets = []
        for r in results:
            if "bbox" not in r or "score" not in r or "class_id" not in r:
                continue
            x1, y1, x2, y2 = r["bbox"]
            score = float(r["score"])
            class_id = int(r["class_id"])
            dets.append([x1, y1, x2, y2, score, class_id])

        if len(dets) == 0:
            return torch.zeros((0, 6), dtype=torch.float32, device=self.device)

        return torch.tensor(dets, dtype=torch.float32, device=self.device)
