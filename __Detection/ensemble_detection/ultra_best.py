# /workspace/__Detection/ensemble_detection/ultra_detector.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from ultralytics import YOLO
from types import SimpleNamespace
import numpy as np
import torch

from .engine.detector_base import DetectorBase
from .engine.registry import register_detector


@register_detector("ultra_best")
class ultraDetector(DetectorBase):
    """
    Ultralytics YOLO 기반 검출기(39클래스 학습)에서
    관심 클래스만 집계용 카테고리로 매핑해 반환합니다.

    반환 포맷:
        SimpleNamespace(
            pred_instances=SimpleNamespace(
                bboxes: FloatTensor[n,4]  # xyxy
                scores: FloatTensor[n]
                labels: LongTensor[n]     # 집계용 카테고리 ID (0, 10..18)
            )
        )

    집계용 카테고리:
      0: worker (사람 계열 전체 묶음: worker/signalman 및 small_* 변형, PPE 미착용 변형 등)
      10~18: 주요 장비류 (excavator, dump_truck, crane_mobile, tower_crane, cargo_truck,
                         forklift, mixer_truck, dozer, scissor_lift)
    """

    DETECTOR_NAME = "ultra_best"  # source 필드용
    DEFAULT_DEVICE = "cuda"
    DEFAULT_CKPT = (
        "/workspace/PretrainedModel_by_JeonYT/new_weights/"
        "250909_pid_147_152_154_155_156_159_160_162_163_164_171_172_173_174_175_176_merged_617장_39cls_yolo11_960/"
        "con8_20250909_061623/weights/best.pt"
    )

    # === YOLO 학습 클래스(39개): YAML 순서와 인덱스 동일해야 함 ===
    DEFAULT_CLASSES: List[str] = [
        "small_worker",                  # 0
        "small_worker_no_vest",          # 1
        "small_worker_no_helmet_no_vest",# 2
        "small_signalman",               # 3
        "small_signalman_no_red",        # 4
        "small_helmet",                  # 5
        "small_head",                    # 6
        "small_harness",                 # 7
        "small_harness_rope",            # 8
        "excavator",                     # 9
        "bucket",                        # 10
        "no_bucket",                     # 11
        "s_bucket",                      # 12
        "dump_truck",                    # 13
        "small_truck",                   # 14
        "tank_lorry",                    # 15
        "cargo_truck",                   # 16
        "mixer_truck",                   # 17
        "crane_mobile",                  # 18
        "tower_crane",                   # 19
        "pile_driver",                   # 20
        "drilling_rig",                  # 21
        "forklift",                      # 22
        "dozer",                         # 23
        "scissor_lift",                  # 24
        "worker",                        # 25
        "worker_no_vest",                # 26
        "worker_no_helmet_no_vest",      # 27
        "signalman",                     # 28
        "signalman_no_red",              # 29
        "helmet",                        # 30
        "head",                          # 31
        "harness",                       # 32
        "harness_rope",                  # 33
        "non_worker",                    # 34
        "non_helmet",                    # 35
        "non_head",                      # 36
        "non_excavator",                 # 37
        "non_dump_truck",                # 38
    ]

    # === 모델 인덱스 → 집계용 카테고리 ID 매핑 ===
    # - 사람 계열(작은/미착용/신호수 변형 포함): 0 (worker)
    # - 차량/장비: 각 표준 ID로
    # - PPE/부정(non_*)/기타 장비(small_truck, tank_lorry, pile_driver, drilling_rig 등)는 미매핑 → 필터
    DEFAULT_ID2COCO: Dict[int, int] = {
        # people family -> 0
        0: 0,   # small_worker
        1: 0,   # small_worker_no_vest
        2: 0,   # small_worker_no_helmet_no_vest
        3: 0,   # small_signalman
        4: 0,   # small_signalman_no_red
        25: 0,  # worker
        26: 0,  # worker_no_vest
        27: 0,  # worker_no_helmet_no_vest
        28: 0,  # signalman
        29: 0,  # signalman_no_red

        # heavy equipment / vehicles
        9:  10,  # excavator
        13: 11,  # dump_truck
        18: 12,  # crane_mobile
        19: 13,  # tower_crane
        16: 14,  # cargo_truck
        22: 15,  # forklift
        17: 16,  # mixer_truck
        23: 17,  # dozer
        24: 18,  # scissor_lift
    }

    # 집계 ID → 고정 라벨 (DetectionAPI.DEFAULT_CLASS_MAP과 합치게 유지)
    DEFAULT_COCO2NAME: Dict[int, str] = {
        0:  "worker",
        10: "excavator",
        11: "dump_truck",
        12: "crane_mobile",
        13: "tower_crane",
        14: "cargo_truck",
        15: "forklift",
        16: "mixer_truck",
        17: "dozer",
        18: "scissor_lift",
    }

    def __init__(
        self,
        device: str = DEFAULT_DEVICE,
        checkpoint: str = DEFAULT_CKPT,
        class_names: Optional[List[str]] = None,
        id2coco_map: Optional[Dict[int, int]] = None,
    ):
        # 클래스/매핑 정의
        self._class_names: List[str] = class_names or self.DEFAULT_CLASSES
        self._id2coco: Dict[int, int] = id2coco_map or self.DEFAULT_ID2COCO
        # 집계 라벨 고정(사람=worker)
        self._coco2name: Dict[int, str] = self.DEFAULT_COCO2NAME.copy()

        # 모델 로드
        self._model = YOLO(checkpoint)
        self._device = device
        print(f"[DEBUG] VehicleDetector loaded with Ultralytics YOLO on {device}")

    # === 필수 프로퍼티 ===
    @property
    def model(self) -> Any:
        return self._model

    @property
    def id2coco(self) -> Dict[int, int]:
        return self._id2coco

    @property
    def coco2name(self) -> Dict[int, str]:
        return self._coco2name

    # === detect() ===
    def detect(self, image: Any):
        """
        Returns:
            SimpleNamespace(
                pred_instances=SimpleNamespace(
                    bboxes: FloatTensor[n,4]  # xyxy
                    scores: FloatTensor[n]
                    labels: LongTensor[n]     # 집계용 카테고리 ID(0,10..18)
                )
            )
        """
        results = self._model.predict(source=image, device=self._device, verbose=False)
        r = results[0]
        boxes = r.boxes

        # YOLO 원본 출력
        all_labels = boxes.cls.cpu().numpy().astype(int)
        bboxes_np  = boxes.xyxy.cpu().numpy()
        scores_np  = boxes.conf.cpu().numpy()

        # id2coco에 등록된 클래스만 통과
        keep_mask = np.array([lbl in self._id2coco for lbl in all_labels], dtype=bool)
        if keep_mask.size == 0 or not keep_mask.any():
            pred_instances = SimpleNamespace(
                bboxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,),    dtype=torch.float32),
                labels=torch.zeros((0,),    dtype=torch.int64),
            )
            return SimpleNamespace(pred_instances=pred_instances)

        kept_labels = all_labels[keep_mask]
        bboxes = torch.as_tensor(bboxes_np[keep_mask], dtype=torch.float32)
        scores = torch.as_tensor(scores_np[keep_mask], dtype=torch.float32)

        # 모델 인덱스 → 집계용 카테고리 ID로 변환
        mapped = [self._id2coco[int(l)] for l in kept_labels]
        labels = torch.as_tensor(mapped, dtype=torch.int64)

        pred_instances = SimpleNamespace(
            bboxes=bboxes,
            scores=scores,
            labels=labels,
        )
        return SimpleNamespace(pred_instances=pred_instances)
