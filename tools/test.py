import cv2, os, time
RTSP = "rtsp://210.99.70.120:1935/live/cctv001.stream"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000|buffer_size;102400"

cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit("열기 실패")

while True:
    ok, f = cap.read()
    if not ok: time.sleep(0.2); continue
    cv2.imshow("RTSP (FFmpeg)", f)
    if cv2.waitKey(1) & 0xFF in (27, ord('q')): break
cap.release(); cv2.destroyAllWindows()
