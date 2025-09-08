import cv2
import numpy as np
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg"}

state = {"frame": None, "points": [], "scale": 1.0}

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        scale = param["scale"]
        # 축소된 좌표 → 원본 좌표 환산
        orig_x, orig_y = int(x / scale), int(y / scale)
        print(f"Clicked pixel position (original): (x={orig_x}, y={orig_y})")
        param["points"].append((orig_x, orig_y))

def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS

def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS

def draw_points(img, points, scale):
    for (px, py) in points:
        sx, sy = int(px * scale), int(py * scale)
        cv2.circle(img, (sx, sy), 4, (0, 255, 0), -1)
        cv2.putText(img, f"({px},{py})", (sx + 8, sy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return img

def imread_unicode(path: str):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def resize_to_fit(img, max_width=1280, max_height=720):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)  # 화면 크기 기준 축소
    if scale < 1.0:
        img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    else:
        img_resized = img
    return img_resized, scale

def main():
    file_path = r"D:\OneDrive - 스마트인사이드에이아이\문서\32.프로젝트\__2023_02_STEAM_한국연구재단(2023.04. - 2027.12.)\12_workspace\SIAI-MCMT_homography solution\250903_실내실험2_성균관대2종합연구동\250904_homograph_coordinate-plane2.jpg"
    p = Path(file_path)

    if not p.exists():
        print("파일을 찾을 수 없습니다:", file_path)
        return

    win = "Viewer"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, click_event, state)

    if is_image(p):
        img = imread_unicode(str(p))
        if img is None:
            print("이미지를 열 수 없습니다.")
            return
        state["frame"] = img
        print("이미지 모드: ESC 종료, 클릭으로 좌표 출력")
        print("원본 크기:", img.shape[1], "x", img.shape[0])

        while True:
            frame_vis, state["scale"] = resize_to_fit(state["frame"])
            frame_vis = draw_points(frame_vis, state["points"], state["scale"])
            cv2.imshow(win, frame_vis)
            if cv2.waitKey(30) & 0xFF == 27:
                break

    elif is_video(p):
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            print("비디오를 열 수 없습니다.")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = max(1, int(1000 / fps)) if fps > 0 else 30

        paused = False
        print("비디오 모드: ESC 종료, SPACE 일시정지/재생")

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("비디오가 끝났습니다.")
                    break
                state["frame"] = frame

            frame_vis, state["scale"] = resize_to_fit(state["frame"])
            frame_vis = draw_points(frame_vis, state["points"], state["scale"])
            cv2.imshow(win, frame_vis)

            key = cv2.waitKey(delay if not paused else 0) & 0xFF
            if key == 27:
                break
            elif key == 32:
                paused = not paused
            elif key in (83, 2555904):  # →
                paused = True
                ret, frame = cap.read()
                if ret:
                    state["frame"] = frame
                else:
                    print("더 이상 스텝 이동할 프레임이 없습니다.")
                    break

        cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
