import cv2
import stream_cctv_cpp

# C++ 기반 객체 생성
cam = stream_cctv_cpp.StreamCCTV("rtsp://210.99.70.120:1935/live/cctv001.stream", maxWidth=960)
cam.start()

while True:
    frame = cam.capture(copy=False)
    if frame.size > 0:
        img = frame.reshape((frame.shape[0], frame.shape[1], 3))
        cv2.imshow("CCTV", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.stop()
cv2.destroyAllWindows()
