import cv2

def resize_image(image_path: str, save_path: str, scale: float = 0.2):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")

    height, width = img.shape[:2]
    new_w, new_h = int(width * scale), int(height * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(save_path, resized)

    return new_w, new_h

# 예시
src = r"D:\__________Workspace__________\video_cutter\downgradrer\asset\plan_origin.png"
dst = r"D:\__________Workspace__________\video_cutter\downgradrer\asset\plan_resized.png"

w, h = resize_image(src, dst, scale=0.2)
print(f"변환된 이미지 크기: {w} x {h}")
