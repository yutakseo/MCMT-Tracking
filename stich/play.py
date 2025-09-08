import cv2

def play_video(video_path: str, scale: float = 1.0, width: int = None, height: int = None):
    """
    영상 재생 함수 (창 크기 조절 가능)

    Parameters
    ----------
    video_path : str
        재생할 영상 파일 경로
    scale : float
        크기 비율 (0.5 = 절반, 2.0 = 2배). width/height가 지정되지 않았을 때만 적용
    width : int
        출력 창의 가로 크기 (픽셀). None이면 자동
    height : int
        출력 창의 세로 크기 (픽셀). None이면 자동
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ 영상을 열 수 없습니다:", video_path)
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("▶ 영상 재생이 끝났습니다.")
            break

        # 원본 크기 가져오기
        h, w = frame.shape[:2]

        if width is not None and height is not None:
            frame = cv2.resize(frame, (width, height))
        elif scale != 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        cv2.imshow("Video Player", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 방법 1: 비율로 크기 조절 (절반 크기)
    # play_video("sample_video.mp4", scale=0.5)

    # 방법 2: 고정 크기 지정 (640x480)
    play_video(r"C:\Users\서유탁\Downloads\plan_projection1.mp4", width=640, height=480)
