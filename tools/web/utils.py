"""
웹 관련 유틸리티 함수들
- 오버레이 관리
- 스트림 관리
- 이미지 처리
"""
from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np
import threading
import os

# ───────────────────────────────────────────────────────────────────────────────
# 오버레이 관리자
# ───────────────────────────────────────────────────────────────────────────────
class WebOverlayManager:
    """바운딩박스/라벨 오버레이 관리"""
    def __init__(self):
        self._overlays: Dict[str, List[Dict[str, Any]]] = {}
        self._class_map: Dict[int, str] = {}
        self._lock = threading.Lock()

    def set_overlays(self, overlays: Dict[str, List[Dict[str, Any]]]):
        with self._lock:
            self._overlays = overlays or {}

    def set_class_map(self, cls_map: Dict[int, str]):
        if isinstance(cls_map, dict):
            self._class_map = {int(k): str(v) for k, v in cls_map.items()}
            print(f"✅ class_map 설정됨: {len(self._class_map)} classes")

    def get_overlay_counts(self) -> Dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self._overlays.items()}

    def get_overlay_details(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            return {k: list(v) for k, v in self._overlays.items()}

    def get_class_map(self) -> Dict[int, str]:
        return self._class_map.copy()

    def get_cam_overlays(self, cam_name: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._overlays.get(cam_name, []))

    def _color_for_id(self, track_id: Optional[int]) -> tuple:
        if track_id is None:
            return (0, 255, 0)
        h = (int(track_id) * 2654435761) & 0xFFFFFFFF
        return (h & 255, (h >> 8) & 255, (h >> 16) & 255)

    def _resolve_label(self, item: Dict[str, Any]) -> str:
        """
        라벨 해석 우선순위:
          1) item['label']이 'unknown/obj'가 아니고 **숫자 문자열이 아니면** 사용
          2) class_id가 있고 _class_map에 있으면 사용
          3) class_id만 있으면 'class_{id}'
          4) 기본 'obj'
        """
        label = item.get("label")
        cid = item.get("class_id", None)

        # 1) 명시 라벨이 있고 'unknown/obj'가 아니며 숫자형이 아닐 때만 사용
        if isinstance(label, str):
            lab = label.strip()
            if lab and lab.lower() not in ["unknown", "obj"]:
                # 숫자 문자열(예: "0", "10")은 무시 → class_map 우선
                is_numeric_like = lab.isdigit()
                if not is_numeric_like:
                    try:
                        float(lab)
                        is_numeric_like = True
                    except Exception:
                        pass
                if not is_numeric_like:
                    if isinstance(cid, (int, np.integer)) and int(cid) >= 0:
                        return f"{lab}({int(cid)})"
                    return lab

        # 2) class_id → class_map
        if isinstance(cid, (int, np.integer)):
            cid_int = int(cid)
            if cid_int >= 0 and cid_int in self._class_map:
                return f"{self._class_map[cid_int]}({cid_int})"
            elif cid_int >= 0:
                return f"class_{cid_int}"

        # 3) 기본
        return "obj"

    def _norm_bbox_to_xyxy(self, bbox, img_w, img_h, it) -> Optional[tuple]:
        if bbox is None:
            return None
        try:
            arr = np.asarray(bbox, dtype=float).reshape(-1)
        except Exception:
            return None
        if arr.size < 4:
            return None

        src_w = it.get("src_w")
        src_h = it.get("src_h")
        if isinstance(src_w, (int, float)) and isinstance(src_h, (int, float)) and src_w > 0 and src_h > 0:
            sx = img_w / float(src_w)
            sy = img_h / float(src_h)
            def scale_fn(x, y, w=None, h=None):
                return (x * sx, y * sy, (w * sx if w is not None else None), (h * sy if h is not None else None))
        else:
            if np.min(arr[:4]) >= 0.0 and np.max(arr[:4]) <= 1.0001:
                def scale_fn(x, y, w=None, h=None):
                    return (x * img_w, y * img_h, (w * img_w if w is not None else None), (h * img_h if h is not None else None))
            else:
                def scale_fn(x, y, w=None, h=None):
                    return (x, y, w, h)

        x1 = y1 = x2 = y2 = None
        fmt = (it.get("format") or "").lower()

        if fmt in ("xyxy", "x1y1x2y2", "tlbr"):
            x1_, y1_, x2_, y2_ = arr[:4]
            x1, y1, _, _ = scale_fn(x1_, y1_)
            x2, y2, _, _ = scale_fn(x2_, y2_)
        elif fmt == "xywh" or fmt == "tlwh":
            x, y, w, h = arr[:4]
            x, y, w, h = scale_fn(x, y, w, h)
            x1, y1, x2, y2 = x, y, x + w, y + h
        elif fmt in ("cxcywh", "center"):
            cx, cy, w, h = arr[:4]
            cx, cy, w, h = scale_fn(cx, cy, w, h)
            x1, y1, x2, y2 = cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0
        else:
            a, b, c, d = arr[:4]
            if c > a and d > b:
                x1_, y1_, x2_, y2_ = a, b, c, d
                x1, y1, _, _ = scale_fn(x1_, y1_)
                x2, y2, _, _ = scale_fn(x2_, y2_)
            else:
                if c >= 0 and d >= 0:
                    x, y, w, h = a, b, c, d
                    x, y, w, h = scale_fn(x, y, w, h)
                    x1, y1, x2, y2 = x, y, x + w, y + h
                else:
                    cx, cy, w, h = a, b, c, d
                    cx, cy, w, h = scale_fn(cx, cy, w, h)
                    x1, y1, x2, y2 = cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0

        if x2 is None or y2 is None or x2 <= x1 or y2 <= y1:
            return None

        x1 = int(max(0, min(round(x1), img_w - 1)))
        y1 = int(max(0, min(round(y1), img_h - 1)))
        x2 = int(max(0, min(round(x2), img_w - 1)))
        y2 = int(max(0, min(round(y2), img_h - 1)))
        return x1, y1, x2, y2

    def draw_overlays(self, img_bgr: np.ndarray, items: List[Dict[str, Any]]) -> np.ndarray:
        H, W = img_bgr.shape[:2]
        print(f"[DEBUG] WebOverlayManager.draw_overlays: processing {len(items)} items on image {W}x{H}")
        print(f"[DEBUG] WebOverlayManager.draw_overlays: class_map = {self._class_map}")

        for i, it in enumerate(items):
            print(f"[DEBUG] WebOverlayManager item {i}: {it}")
            xyxy = self._norm_bbox_to_xyxy(it.get("bbox"), W, H, it)
            if xyxy is None:
                print(f"[DEBUG] WebOverlayManager item {i}: bbox normalization failed")
                continue

            x1, y1, x2, y2 = xyxy
            print(f"[DEBUG] WebOverlayManager item {i}: normalized bbox = ({x1}, {y1}, {x2}, {y2})")
            if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0 or x2 >= W or y2 >= H:
                print(f"[DEBUG] WebOverlayManager item {i}: invalid bbox coordinates")
                continue

            tid = it.get("track_id")
            color = self._color_for_id(tid)

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)

            text_parts = []
            if tid is not None:
                text_parts.append(f"ID:{tid}")
            label = self._resolve_label(it)
            if label and label != "obj":
                text_parts.append(label)
            score = it.get("score", None)
            if isinstance(score, (int, float)):
                text_parts.append(f"{score:.2f}")
            text = " ".join(text_parts) if text_parts else "obj"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx1, ty1 = x1, max(0, y1 - (th + 8))
            tx2, ty2 = x1 + tw + 8, y1
            cv2.rectangle(img_bgr, (tx1, ty1), (tx2, ty2), color, -1)
            cv2.putText(img_bgr, text, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            print(f"[DEBUG] WebOverlayManager item {i}: drawn bbox and label '{text}'")
        return img_bgr

# ───────────────────────────────────────────────────────────────────────────────
# 스트림 관리자
# ───────────────────────────────────────────────────────────────────────────────
class WebStreamManager:
    """카메라 스트림 관리"""
    def __init__(self):
        self._streams: Dict[str, Any] = {}
        self._JPEG_ATTRS = ("_jpg", "jpg", "last_jpeg", "jpeg", "jpeg_bytes")
        self._BGR_ATTRS  = ("bgr", "frame", "last_frame", "image", "_frame")

    def set_streams(self, streams: Dict[str, Any]):
        self._streams = streams or {}

    def get_streams(self) -> Dict[str, Any]:
        return self._streams.copy()

    def get_stream(self, name: str) -> Optional[Any]:
        return self._streams.get(name)

    def jpeg_from_ndarray(self, img: np.ndarray, quality: int = 85) -> bytes:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else b""

    def fallback_plan_frame(self) -> bytes:
        PLAN_IMG = "/workspace/assets/250904_homograph_coordinate-plane2.jpg"
        if os.path.exists(PLAN_IMG):
            img = cv2.imread(PLAN_IMG)
            if img is not None:
                return self.jpeg_from_ndarray(img)
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(black, "No Plan Image", (150, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
        return self.jpeg_from_ndarray(black)

    def extract_frame_pair(self, stream_obj: Any) -> Tuple[Optional[np.ndarray], Optional[bytes]]:
        for attr in self._JPEG_ATTRS:
            data = getattr(stream_obj, attr, None)
            if isinstance(data, (bytes, bytearray)) and len(data) > 100:
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    return img, bytes(data)
        for attr in self._BGR_ATTRS:
            img = getattr(stream_obj, attr, None)
            if isinstance(img, np.ndarray) and img.ndim >= 2 and img.size > 0:
                return img, None
        return None, None
