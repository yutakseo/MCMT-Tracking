# /workspace/tools/web/server.py
"""
FastAPI 웹 서버
- 메인 웹 애플리케이션
- 스트림 엔드포인트
- 디버그 엔드포인트
"""
from typing import Optional, Dict, Any, List
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, JSONResponse
import sys
import cv2
import numpy as np
import time
import asyncio

sys.path.append("/workspace")

from .utils import WebOverlayManager, WebStreamManager
from .planviz import PlanViz

app = FastAPI(title="Multi-Camera Web Server", version="1.4.0")

# 전역 핸들
webviz: Optional[PlanViz] = None
overlay_manager = WebOverlayManager()
stream_manager = WebStreamManager()

CAM_IMG_DEFAULT_SCALE = 0.6

# ───────────────────────────────────────────────────────────────────────────────
# 주입 API
# ───────────────────────────────────────────────────────────────────────────────
def set_webviz(viz: PlanViz):
    global webviz
    webviz = viz
    print("✅ WebPlanViz가 웹 서버에 연결되었습니다")

def set_cam_streams(streams: Dict[str, Any]):
    stream_manager.set_streams(streams)
    print(f"✅ {len(streams)}개 카메라 스트림이 웹 서버에 연결되었습니다")

def set_cam_overlays(overlays: Dict[str, List[Dict[str, Any]]]):
    # 디버그 로그(요약)
    print("[DEBUG] set_cam_overlays 호출...")
    for k, v in overlays.items():
        print(f"[DEBUG] set_cam_overlays: {k} = {len(v)} items")
    overlay_manager.set_overlays(overlays)
    print("[DEBUG] set_cam_overlays: 오버레이 설정 완료")

def set_class_map(cls_map: Dict[int, str]):
    overlay_manager.set_class_map(cls_map)
    print(f"✅ class_map 주입됨: {len(cls_map)} classes")

# ───────────────────────────────────────────────────────────────────────────────
# 헬스/디버그
# ───────────────────────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok", status_code=200)

@app.get("/debug")
async def debug():
    stream_info = {}
    ov_counts = overlay_manager.get_overlay_counts()
    overlay_details = overlay_manager.get_overlay_details()
    for name, stream in stream_manager.get_streams().items():
        info = {"has_frame": False, "frame_type": None, "frame_size": 0,
                "is_alive": False, "overlays": ov_counts.get(name, 0)}
        try:
            img, jpg = stream_manager.extract_frame_pair(stream)
            if img is not None:
                info["has_frame"] = True
                info["frame_type"] = "bgr"
                info["frame_size"] = int(img.size)
                info["frame_shape"] = list(img.shape)
            elif jpg is not None:
                info["has_frame"] = True
                info["frame_type"] = "jpeg"
                info["frame_size"] = len(jpg)
        except Exception as e:
            info["error"] = str(e)
        try:
            th = getattr(stream, "_th", None)
            info["is_alive"] = bool(th and th.is_alive())
        except Exception:
            pass
        stream_info[name] = info

    return {
        "webviz_connected": webviz is not None,
        "cam_streams": list(stream_manager.get_streams().keys()),
        "cam_stream_count": len(stream_manager.get_streams()),
        "stream_info": stream_info,
        "overlay_keys": list(ov_counts.keys()),
        "overlay_counts": ov_counts,
        "overlay_details": overlay_details,
        "class_map": overlay_manager.get_class_map(),
        "class_map_size": len(overlay_manager.get_class_map()),
        "timestamp": time.time(),
    }

@app.get("/overlay_keys")
async def overlay_keys():
    return JSONResponse({
        "stream_keys": list(stream_manager.get_streams().keys()),
        "overlay_keys": list(overlay_manager.get_overlay_counts().keys()),
        "overlay_counts": overlay_manager.get_overlay_counts(),
        "class_map_size": len(overlay_manager.get_class_map()),
    })

# ───────────────────────────────────────────────────────────────────────────────
# 메인 페이지
# ───────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    streams = stream_manager.get_streams()
    class_map = overlay_manager.get_class_map()
    if streams:
        cam_cards = "".join(
            f"""
            <div class="cam-card">
              <div class="cam-label">{name}</div>
              <img class="cam-img" src="/cam/{name}?scale={CAM_IMG_DEFAULT_SCALE}" alt="{name}"/>
            </div>
            """
            for name in streams.keys()
        )
    else:
        cam_cards = """
        <div class="cam-empty">
          <p>카메라 스트림이 연결되지 않았습니다. main.py 실행 또는 set_cam_streams 주입을 확인하세요.</p>
        </div>
        """

    return f"""
    <html>
      <head>
        <meta charset="utf-8">
        <title>Multi-Camera Object Tracking System</title>
        <style>
          :root {{
            --bg:#111; --fg:#eee; --card:#1b1b1b; --accent:#00ff00; --border:#333;
          }}
          * {{ box-sizing: border-box; }}
          body {{
            background: var(--bg);
            color: var(--fg);
            font-family: system-ui, Arial, sans-serif;
            margin: 20px;
          }}
          h1 {{ margin: 0 0 16px; }}
          .refresh-btn {{
            background: var(--accent); color:#000; border:none; padding:10px 20px;
            border-radius:6px; cursor:pointer; font-weight:bold; margin: 10px 0 20px;
          }}
          .refresh-btn:hover {{ filter: brightness(0.9); }}

          .plan-wrap {{
            width: 100%;
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 16px;
          }}
          .plan-title {{ margin: 0 0 10px; font-size: 18px; }}
          .plan-img {{
            width: 100%; height: auto;
            border: 2px solid var(--border); border-radius: 8px;
            display: block; background: #000;
          }}

          .cams-row {{
            display: grid;
            grid-template-columns: repeat(3, minmax(260px, 1fr));
            gap: 12px; width: 100%;
          }}
          .cam-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px;
          }}
          .cam-label {{
            color: var(--accent);
            font-weight: bold;
            margin-bottom: 8px;
          }}
          .cam-img {{
            width: 100%;
            height: auto;
            object-fit: contain;
            border: 2px solid var(--accent);
            border-radius: 8px;
            background: #000;
            display: block;
          }}
          .cam-empty {{
            grid-column: 1 / -1;
            background: #2a2a2a;
            border: 1px dashed var(--border);
            padding: 20px; border-radius: 10px;
          }}

          .status-box {{
            margin-top: 20px; padding: 15px; background:#222; border-radius: 8px;
            border: 1px solid var(--border);
          }}

          @media (max-width: 1100px) {{
            .cams-row {{ grid-template-columns: repeat(2, minmax(240px, 1fr)); }}
          }}
          @media (max-width: 720px) {{
            .cams-row {{ grid-template-columns: 1fr; }}
          }}
        </style>
        <script>
          function refreshPage() {{ location.reload(); }}
          setInterval(refreshPage, 30000);
        </script>
      </head>
      <body>
        <h1>🎯 Multi-Camera Object Tracking System</h1>
        <button class="refresh-btn" onclick="refreshPage()">🔄 새로고침</button>

        <section class="plan-wrap">
          <h2 class="plan-title">📊 Plan View (Fused Objects)</h2>
          <img class="plan-img" src="/stream_plan" alt="Plan View"/>
        </section>

        <section class="cams-row">
          {cam_cards}
        </section>

        <div class="status-box">
          <h3>📈 System Status</h3>
          <p>• <strong>Multi-Camera System:</strong> shared model</p>
          <p>• <strong>Real-time:</strong> Detection → Tracking → Homography</p>
          <p>• <strong>Connected Streams:</strong> {len(streams)} cameras</p>
          <p>• <strong>Classes:</strong> {len(class_map)}</p>
          <p>• <strong>Debug:</strong> <a href="/debug" target="_blank" style="color: var(--accent);">Debug Info</a> | <a href="/overlay_keys" target="_blank" style="color: var(--accent);">Overlay Keys</a></p>
        </div>
      </body>
    </html>
    """

# ───────────────────────────────────────────────────────────────────────────────
# 스트림 엔드포인트
# ───────────────────────────────────────────────────────────────────────────────
@app.get("/stream_plan")
async def stream_plan():
    boundary = "frame"
    async def agen():
        while True:
            try:
                if webviz is not None:
                    frame = webviz.peek_jpeg()
                else:
                    frame = stream_manager.fallback_plan_frame()
                yield (b"--" + boundary.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"[stream_plan] error: {e}")
                await asyncio.sleep(0.5)
    return StreamingResponse(agen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

@app.get("/cam/{cam_name}")
async def stream_cam(cam_name: str, scale: float = 1.0):
    boundary = "frame"
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    scale = max(0.1, min(scale, 1.0))

    async def agen():
        while True:
            try:
                stream = stream_manager.get_stream(cam_name)
                if stream is None:
                    wait_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(wait_img, f"No stream: {cam_name}", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    out_img = wait_img
                    if scale != 1.0:
                        h, w = out_img.shape[:2]
                        out_img = cv2.resize(out_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                    out_bytes = stream_manager.jpeg_from_ndarray(out_img)
                else:
                    img_bgr, jpg_bytes = stream_manager.extract_frame_pair(stream)
                    if img_bgr is None and jpg_bytes is None:
                        wait_img = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(wait_img, f"Waiting stream: {cam_name}", (50, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        out_img = wait_img
                        if scale != 1.0:
                            h, w = out_img.shape[:2]
                            out_img = cv2.resize(out_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                        out_bytes = stream_manager.jpeg_from_ndarray(out_img)
                    else:
                        if img_bgr is None and jpg_bytes is not None:
                            nparr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        overlays = overlay_manager.get_cam_overlays(cam_name)
                        if overlays:
                            img_bgr = overlay_manager.draw_overlays(img_bgr, overlays)

                        if scale != 1.0:
                            h, w = img_bgr.shape[:2]
                            img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

                        out_bytes = stream_manager.jpeg_from_ndarray(img_bgr)

                yield (b"--" + boundary.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + out_bytes + b"\r\n")
                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"[stream_cam:{cam_name}] error: {e}")
                await asyncio.sleep(0.5)
    return StreamingResponse(agen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

# 단독 실행용
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
