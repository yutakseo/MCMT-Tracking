# /workspace/__Detection/tools/show.py
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
except ImportError:
    torch = None


def show_detections(image_or_path, detections_result, score_thr: float = 0.3,
                    save_path="/workspace/results",show:bool=True):
    """
    탐지 결과 시각화 + 저장 + matplotlib 표시

    Args:
        image_or_path: str (이미지 경로) or np.ndarray
        detections: (N,5|6) ndarray / torch.Tensor / list of dict
        score_thr: 최소 confidence
        save_path: None이면 저장 안함 / str이면 경로
        show: True → matplotlib으로 표시
    """
    # 0) 이미지 로드
    if isinstance(image_or_path, str):
        img = cv2.imread(image_or_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_or_path}")
        in_name = os.path.splitext(os.path.basename(image_or_path))[0]
    else:
        img = image_or_path.copy()
        in_name = "image"

    vis_img = img.copy()

    # 1) detections 표준화
    def _normalize_detections(detections):
        if torch is not None and isinstance(detections, torch.Tensor):
            detections = detections.detach().cpu().numpy()
        if isinstance(detections, list) and len(detections) > 0 and isinstance(detections[0], dict):
            rows = []
            for d in detections:
                x1, y1, x2, y2 = d["bbox"][:4]
                score = d.get("score", 0.0)
                cls_id = d.get("class_id", -1)
                rows.append([x1, y1, x2, y2, score, cls_id])
            return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 6), np.float32)
        if isinstance(detections, np.ndarray):
            dets = detections.astype(np.float32)
            if dets.shape[1] == 5:
                pad = np.full((dets.shape[0], 1), -1, dtype=np.float32)
                dets = np.concatenate([dets, pad], axis=1)
            return dets
        return np.zeros((0, 6), dtype=np.float32)

    dets_np = _normalize_detections(detections_result)

    # 2) draw
    for row in dets_np:
        x1, y1, x2, y2, score = row[:5]
        if score < score_thr:
            continue
        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 0, 255), 2)
        cv2.putText(vis_img, f"{score:.2f}", (int(x1), max(0, int(y1) - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 3) 저장
    if save_path:
        if os.path.isdir(save_path) or save_path.endswith(os.path.sep):
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f"{in_name}_vis.jpg")
        else:
            out_dir = os.path.dirname(save_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            root, ext = os.path.splitext(save_path)
            save_file = save_path if ext.lower() in [".jpg", ".jpeg", ".png", ".bmp"] else root + ".jpg"

        ok = cv2.imwrite(save_file, vis_img)
        if not ok:
            raise RuntimeError(f"Failed to write image to: {save_file}")
        print(f"✅ 저장 완료: {save_file}")

    # 4) matplotlib으로 표시 (컨테이너 환경에서 유용)
    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

    return vis_img
