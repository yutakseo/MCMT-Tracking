# /workspace/app_web.py
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from tools.webviz import WebPlanViz

app = FastAPI()

# 전역 핸들(런타임에 주입)
webviz: Optional[WebPlanViz] = None
_cam_streams: Dict[str, Any] = {}  # {"cam1": CamMJPEG, ...}

@app.get("/", response_class=HTMLResponse)
def index():
    cam_imgs = ""
    if _cam_streams:
        cam_imgs = "<h3>RTSP Cameras</h3><div style='display:flex;gap:12px;flex-wrap:wrap'>"
        for name in _cam_streams.keys():
            cam_imgs += (
                f"<div style='display:flex;flex-direction:column;align-items:center'>"
                f"<div style='color:#bbb;margin-bottom:6px'>{name}</div>"
                f"<img src='/cam/{name}' style='max-width:480px;border:1px solid #333'/>"
                f"</div>"
            )
        cam_imgs += "</div>"

    return f"""
    <html>
      <head><meta charset="utf-8"><title>Fused Plan View</title></head>
      <body style="background:#111;color:#eee;font-family:system-ui,Arial,sans-serif;margin:20px">
        <h2>Fused Plan View</h2>
        <img src="/stream_plan" style="max-width:100%;border:1px solid #333"/>
        {cam_imgs}
      </body>
    </html>
    """

@app.get("/healthz", response_class=HTMLResponse)
def healthz():
    return "ok"

@app.get("/stream_plan")
def stream_plan():
    if webviz is None:
        raise HTTPException(status_code=500, detail="webviz not initialized")
    return StreamingResponse(
        webviz.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/cam/{name}")
def stream_cam(name: str):
    if not _cam_streams or name not in _cam_streams:
        raise HTTPException(status_code=404, detail=f"camera '{name}' not found")
    cam = _cam_streams[name]
    if not hasattr(cam, "mjpeg_generator"):
        raise HTTPException(status_code=500, detail=f"camera '{name}' has no mjpeg_generator()")
    return StreamingResponse(
        cam.mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

def serve_webviz(viz: WebPlanViz, streams: Optional[Dict[str, Any]] = None,
                 host: str = "0.0.0.0", port: int = 8000):
    """
    웹서버 실행. 도면 스트림(/stream_plan) + 카메라 MJPEG(/cam/<name>) 제공.
    - viz.update(pkt) 로 외부에서 프레임 갱신
    - streams: {"cam1": CamMJPEG(...).start(), ...}
    """
    global webviz, _cam_streams
    webviz = viz
    _cam_streams = streams or {}
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
