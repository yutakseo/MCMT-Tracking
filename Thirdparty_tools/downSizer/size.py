import cv2

def get_image_size_cv2(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
    height, width = img.shape[:2]
    return width, height

# 예시
w, h = get_image_size_cv2(r"D:\__________Workspace__________\video_cutter\downgradrer\project_downsize.png")
print(f"이미지 크기: {w} x {h}")
