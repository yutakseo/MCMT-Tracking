import os, time, argparse, cv2

# 지연/끊김 완화 (TCP, 버퍼 줄이기)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000|buffer_size;102400"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("VIDEO_SRC", "rtsp://user:pass@CAM_IP:554/stream"),
                    help="RTSP URL (예: rtsp://user:pass@192.168.0.50:554/stream1)")
    ap.add_argument("--max-width", type=int, default=int(os.environ.get("MAX_WIDTH", "1280")),
                    help="가로 최대폭 (리사이즈로 부하 감소)")
    ap.add_argument("--window", default="CCTV", help="윈도우 제목")
    ap.add_argument("--reconnect-frames", type=int, default=60,
                    help="연속 실패 프레임 수 초과 시 재연결 (기본 60)")
    return ap.parse_args()

def open_cap(rtsp_url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 큐 최소화 → 지연 감소
    return cap

def maybe_resize(frame, max_w: int):
    h, w = frame.shape[:2]
    if w <= max_w: return frame
    scale = max_w / float(w)
    return cv2.resize(frame, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

def main():
    args = parse_args()
    print(f"[INFO] opening: {args.url}")
    cap = open_cap(args.url)

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(args.window, 960, 540)  # 필요하면 창 기본 크기 지정

    bad = 0
    t0, frames = time.time(), 0
    show_fps = True

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            bad += 1
            if bad > args.reconnect_frames:
                print("[WARN] read fail. reconnecting...")
                cap.release()
                time.sleep(1)
                cap = open_cap(args.url)
                bad = 0
            # 실패해도 loop 유지
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')): break
            continue

        bad = 0
        frame = maybe_resize(frame, args.max_width)

        # FPS 계산 표시
        if show_fps:
            frames += 1
            dt = time.time() - t0
            if dt >= 1.0:
                fps = frames / dt
                t0, frames = time.time(), 0
            # 최근 fps를 그리기 위해 계산 주기 외엔 기존 fps 유지
            try:
                fps  # 이전 루프에서 만든 변수가 있으면 사용
            except NameError:
                fps = 0.0
            cv2.putText(frame, f"FPS: {fps:.1f}", (16, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow(args.window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord('p'):      # 일시정지/해제
            while True:
                if cv2.waitKey(30) & 0xFF in (27, ord('q'), ord('p')):
                    break
        elif key == ord('f'):      # FPS 표시 토글
            show_fps = not show_fps

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
