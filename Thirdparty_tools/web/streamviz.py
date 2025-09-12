# /workspace/tools/web/streamviz.py
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

class StreamViz:
    def __init__(
        self,
        bbox_thickness: int = 3,
        font_scale: float = 0.6,
        font_thickness: int = 2,
        show_track_id: bool = True,
        show_class_label: bool = True,
        show_confidence: bool = True,
    ):
        self.bbox_thickness = bbox_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.show_track_id = show_track_id
        self.show_class_label = show_class_label
        self.show_confidence = show_confidence
        self.colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
        ]

    def _get_color_for_id(self, track_id: Optional[int]) -> Tuple[int, int, int]:
        if track_id is None:
            return self.colors[0]
        return self.colors[int(track_id) % len(self.colors)]

    def _resolve_label(self, item: Dict[str, Any], class_map: Dict[int, str]) -> str:
        label = item.get("label")
        class_id = item.get("class_id", None)
        if isinstance(label, str) and label.strip() and label.lower() not in ["unknown", "obj"]:
            if isinstance(class_id, (int, np.integer)) and int(class_id) >= 0:
                return f"{label}({int(class_id)})"
            return label
        if isinstance(class_id, (int, np.integer)):
            class_id_int = int(class_id)
            if class_id_int >= 0 and class_id_int in class_map:
                return f"{class_map[class_id_int]}({class_id_int})"
            elif class_id_int >= 0:
                return f"class_{class_id_int}"
        return "obj"

    def _create_label_text(self, item: Dict[str, Any], class_map: Dict[int, str]) -> str:
        parts = []
        if self.show_track_id:
            tid = item.get("track_id")
            if tid is not None:
                parts.append(f"ID:{tid}")
        if self.show_class_label:
            lab = self._resolve_label(item, class_map)
            if lab and lab != "obj":
                parts.append(lab)
        if self.show_confidence:
            score = item.get("score", None)
            if isinstance(score, (int, float)):
                parts.append(f"{score:.2f}")
        return " ".join(parts) if parts else "obj"

    def draw_overlays(self, img_bgr: np.ndarray, items: List[Dict[str, Any]], class_map: Dict[int, str] = None) -> np.ndarray:
        if class_map is None:
            class_map = {}
        H, W = img_bgr.shape[:2]
        for it in items:
            bbox = it.get("bbox")
            if bbox is None:
                continue
            try:
                x1, y1, x2, y2 = map(int, bbox[:4])
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 >= W or y2 >= H:
                continue
            color = self._get_color_for_id(it.get("track_id"))
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, self.bbox_thickness)
            text = self._create_label_text(it, class_map)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)
            tx1, ty1 = x1, max(0, y1 - (th + 8))
            tx2, ty2 = x1 + tw + 8, y1
            cv2.rectangle(img_bgr, (tx1, ty1), (tx2, ty2), color, -1)
            cv2.putText(img_bgr, text, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, (255, 255, 255),
                        self.font_thickness, cv2.LINE_AA)
        return img_bgr

    def draw_single_object(self, img_bgr: np.ndarray, item: Dict[str, Any], class_map: Dict[int, str] = None) -> np.ndarray:
        return self.draw_overlays(img_bgr, [item], class_map)
