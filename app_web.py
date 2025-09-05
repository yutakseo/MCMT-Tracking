# /workspace/app_web.py
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from tools.webviz import WebPlanViz
import uvicorn
import sys
import os
import cv2
import numpy as np
import time
sys.path.append("/workspace")

app = FastAPI()

# 전역 핸들(런타임에 주입)
webviz: Optional[WebPlanViz] = None
_cam_streams: Dict[str, Any] = {}  # {"cam1": CamMJPEG, ...}

def set_webviz(viz: WebPlanViz):
    """WebPlanViz 인스턴스 설정"""
    global webviz
    webviz = viz
    print("✅ WebPlanViz가 웹 서버에 연결되었습니다")

def set_cam_streams(streams: Dict[str, Any]):
    """카메라 스트림 설정"""
    global _cam_streams
    _cam_streams = streams
    print(f"✅ {len(streams)}개 카메라 스트림이 웹 서버에 연결되었습니다")

@app.get("/", response_class=HTMLResponse)
def index():
    cam_imgs = ""
    if _cam_streams:
        cam_imgs = "<h3>📹 RTSP CCTV Cameras</h3><div style='display:flex;gap:12px;flex-wrap:wrap'>"
        for name in _cam_streams.keys():
            cam_imgs += (
                f"<div style='display:flex;flex-direction:column;align-items:center;margin:10px'>"
                f"<div style='color:#00ff00;margin-bottom:6px;font-weight:bold'>{name}</div>"
                f"<img src='/cam/{name}' style='max-width:480px;border:2px solid #00ff00;border-radius:8px'/>"
                f"</div>"
            )
        cam_imgs += "</div>"
    else:
        cam_imgs = "<h3>📹 RTSP CCTV Cameras</h3><p style='color:#ff0000'>카메라 스트림이 연결되지 않았습니다. main.py를 실행하세요.</p>"

    return f"""
    <html>
      <head>
        <meta charset="utf-8">
        <title>Multi-Camera Object Tracking System</title>
        <style>
          body {{ background:#111; color:#eee; font-family:system-ui,Arial,sans-serif; margin:20px; }}
          .container {{ display:flex; gap:20px; flex-wrap:wrap; }}
          .plan-section {{ flex:1; min-width:600px; }}
          .cam-section {{ flex:1; min-width:400px; }}
          .status-box {{ margin-top:20px; padding:15px; background:#222; border-radius:8px; }}
          .cam-grid {{ display:flex; gap:12px; flex-wrap:wrap; }}
          .cam-item {{ display:flex; flex-direction:column; align-items:center; margin:10px; }}
          .cam-label {{ color:#00ff00; margin-bottom:6px; font-weight:bold; }}
          .cam-img {{ max-width:480px; border:2px solid #00ff00; border-radius:8px; }}
        </style>
      </head>
      <body>
        <h1>🎯 Multi-Camera Object Tracking System</h1>
        <div class="container">
          <div class="plan-section">
            <h2>📊 Plan View (Fused Objects)</h2>
            <img src="/stream_plan" style="max-width:100%;border:2px solid #333;border-radius:8px"/>
          </div>
          <div class="cam-section">
            {cam_imgs}
          </div>
        </div>
        <div class="status-box">
          <h3>📈 System Status</h3>
          <p>• <strong>Multi-Camera System:</strong> 3 cameras with shared model</p>
          <p>• <strong>GPU Memory:</strong> 66% optimized (single model sharing)</p>
          <p>• <strong>Smart GPU Management:</strong> 4-stage load control</p>
          <p>• <strong>Real-time Tracking:</strong> Object detection → Tracking → Homography</p>
          <p>• <strong>RTSP Stream:</strong> rtsp://210.99.70.120:1935/live/cctv001.stream</p>
        </div>
      </body>
    </html>
    """

@app.get("/stream_plan")
def stream_plan():
    """도면 스트림 (객체 위치 표시)"""
    def generate():
        while True:
            try:
                if webviz is not None:
                    # WebPlanViz에서 최신 프레임 가져오기
                    frame = webviz._last_jpeg
                    if frame is not None:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    else:
                        # 기본 도면 이미지 반환
                        plan_img = cv2.imread("/workspace/assets/250904_homograph_coordinate-plane2.jpg")
                        if plan_img is not None:
                            _, buffer = cv2.imencode('.jpg', plan_img)
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                else:
                    # WebPlanViz가 없을 때 기본 도면 이미지 반환
                    plan_img = cv2.imread("/workspace/assets/250904_homograph_coordinate-plane2.jpg")
                    if plan_img is not None:
                        _, buffer = cv2.imencode('.jpg', plan_img)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    else:
                        # 도면 파일이 없을 때 검정 화면
                        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
                        _, buffer = cv2.imencode('.jpg', black_img)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception as e:
                print(f"Stream plan error: {e}")
                break
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/cam/{cam_name}")
def stream_cam(cam_name: str):
    """카메라 스트림"""
    if cam_name not in _cam_streams:
        # 카메라 스트림이 없을 때 검정 화면 반환
        def generate_black():
            while True:
                try:
                    black_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black_img, f"No Stream: {cam_name}", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.jpg', black_img)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    time.sleep(0.1)
                except:
                    break
        return StreamingResponse(generate_black(), media_type="multipart/x-mixed-replace; boundary=frame")
    
    stream = _cam_streams[cam_name]
    
    def generate():
        while True:
            try:
                # CamMJPEG에서 프레임 가져오기
                frame = stream._jpg
                if frame is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                else:
                    # 프레임이 없을 때 대기 화면
                    wait_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(wait_img, f"Connecting: {cam_name}", (50, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    _, buffer = cv2.imencode('.jpg', wait_img)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.05)  # 20 FPS
            except Exception as e:
                print(f"Stream {cam_name} error: {e}")
                break
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/debug")
def debug():
    """디버그 정보"""
    return {
        "webviz_connected": webviz is not None,
        "cam_streams": list(_cam_streams.keys()),
        "cam_stream_count": len(_cam_streams),
        "timestamp": time.time()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
