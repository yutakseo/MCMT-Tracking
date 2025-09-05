from typing import Optional, Dict, Any, List
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse
from tools.webviz import WebPlanViz
import uvicorn
import sys
import os
import cv2
import numpy as np
import time
import asyncio
import threading

sys.path.append("/workspace")

app = FastAPI(title="Multi-Camera Web Server", version="1.2.0")

# ───────────────────────────────────────────────────────────────────────────────
# 전역 핸들 (main.py에서 주입)
# ───────────────────────────────────────────────────────────────────────────────
webviz: Optional[WebPlanViz] = None
_cam_streams: Dict[str, Any] = {}  # {"cam1": CamMJPEG, ...}
PLAN_IMG = "/workspace/assets/250904_homograph_coordinate-plane2.jpg"

# 바운딩박스/라벨 오버레이 저장소
_cam_overlays: Dict[str, List[Dict[str, Any]]] = {}  # {"cam1":[{bbox:[x1,y1,x2,y2], label, score, track_id}], ...}
_cam_ov_lock = threading.Lock()

def set_webviz(viz: WebPlanViz):
    """WebPlanViz 인스턴스 설정 (메인 프로세스에서 주입 필요)"""
    global webviz
    webviz = viz
    print("✅ WebPlanViz가 웹 서버에 연결되었습니다")

def set_cam_streams(streams: Dict[str, Any]):
    """카메라 스트림 설정 (메인 프로세스에서 주입 필요)"""
    global _cam_streams
    _cam_streams = streams or {}
    print(f"✅ {len(_cam_streams)}개 카메라 스트림이 웹 서버에 연결되었습니다")

def set_cam_overlays(overlays: Dict[str, List[Dict[str, Any]]]):
    """카메라별 바운딩박스/라벨 오버레이 주입 (main.py에서 주입)"""
    global _cam_overlays
    with _cam_ov_lock:
        _cam_overlays = overlays or {}

def _get_cam_overlays(cam_name: str) -> List[Dict[str, Any]]:
    with _cam_ov_lock:
        return list(_cam_overlays.get(cam_name, []))

# ───────────────────────────────────────────────────────────────────────────────
# 유틸
# ───────────────────────────────────────────────────────────────────────────────
def _jpeg_from_ndarray(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""

def _fallback_plan_frame() -> bytes:
    if os.path.exists(PLAN_IMG):
        img = cv2.imread(PLAN_IMG)
        if img is not None:
            return _jpeg_from_ndarray(img)
    # 파일 없으면 검정 화면
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(black, "No Plan Image", (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
    return _jpeg_from_ndarray(black)

def _color_for_id(track_id: Optional[int]) -> tuple:
    """트랙ID기반 고정 색상(BGR). ID 없으면 기본 녹색."""
    if track_id is None:
        return (0, 255, 0)
    h = (int(track_id) * 2654435761) & 0xFFFFFFFF
    return (h & 255, (h >> 8) & 255, (h >> 16) & 255)

def _draw_overlays(img_bgr: np.ndarray, items: List[Dict[str, Any]]) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    for it in items:
        bbox = it.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            continue
        x1 = max(0, min(int(x1), W - 1))
        y1 = max(0, min(int(y1), H - 1))
        x2 = max(0, min(int(x2), W - 1))
        y2 = max(0, min(int(y2), H - 1))
        tid = it.get("track_id")
        color = _color_for_id(tid)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        label = it.get("label", "obj")
        score = it.get("score", None)
        tid_txt = f" #{tid}" if tid is not None else ""
        s_txt = f" {score:.2f}" if isinstance(score, (int, float)) else ""
        text = f"{label}{s_txt}{tid_txt}"
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx1, ty1 = x1, max(0, y1 - (th + 6))
        tx2, ty2 = x1 + tw + 6, y1
        cv2.rectangle(img_bgr, (tx1, ty1), (tx2, ty2), color, -1)
        cv2.putText(img_bgr, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img_bgr

# ───────────────────────────────────────────────────────────────────────────────
# 헬스/디버그
# ───────────────────────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok", status_code=200)

@app.get("/debug")
async def debug():
    stream_info = {}
    with _cam_ov_lock:
        ov_counts = {k: len(v) for k, v in _cam_overlays.items()}
    for name, stream in _cam_streams.items():
        info = {"has_frame": False, "frame_size": 0, "is_alive": False, "overlays": ov_counts.get(name, 0)}
        try:
            jpg = getattr(stream, "_jpg", None)
            info["has_frame"] = jpg is not None
            info["frame_size"] = len(jpg) if jpg else 0
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
        "cam_streams": list(_cam_streams.keys()),
        "cam_stream_count": len(_cam_streams),
        "stream_info": stream_info,
        "timestamp": time.time(),
    }

# ───────────────────────────────────────────────────────────────────────────────
# 메인 페이지 (도면 위, 카메라 3열 아래)
# ───────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    if _cam_streams:
        cam_cards = "".join(
            f"""
            <div class="cam-card">
              <div class="cam-label">{name}</div>
              <img class="cam-img" src="/cam/{name}" alt="{name}"/>
            </div>
            """
            for name in _cam_streams.keys()
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
            height: 260px;
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
            .cam-img {{ height: 220px; }}
          }}
          @media (max-width: 720px) {{
            .cams-row {{ grid-template-columns: 1fr; }}
            .cam-img {{ height: 200px; }}
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
          <p>• <strong>Connected Streams:</strong> {len(_cam_streams)} cameras</p>
        </div>
      </body>
    </html>
    """

# ───────────────────────────────────────────────────────────────────────────────
# 스트림 엔드포인트 (비동기 제너레이터 사용)
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
                    frame = _fallback_plan_frame()
                yield (b"--" + boundary.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"[stream_plan] error: {e}")
                await asyncio.sleep(0.5)
    return StreamingResponse(agen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

@app.get("/cam/{cam_name}")
async def stream_cam(cam_name: str):
    boundary = "frame"
    async def agen():
        while True:
            try:
                stream = _cam_streams.get(cam_name)
                frame = getattr(stream, "_jpg", None) if stream is not None else None

                if frame is None or len(frame) < 1000:
                    wait_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(wait_img, f"Connecting: {cam_name}", (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.putText(wait_img, "Please wait...", (50, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                    out_bytes = _jpeg_from_ndarray(wait_img)
                else:
                    overlays = _get_cam_overlays(cam_name)
                    if overlays:
                        nparr = np.frombuffer(frame, dtype=np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            img = _draw_overlays(img, overlays)
                            out_bytes = _jpeg_from_ndarray(img)
                        else:
                            out_bytes = frame
                    else:
                        out_bytes = frame

                yield (b"--" + boundary.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + out_bytes + b"\r\n")
                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"[stream_cam:{cam_name}] error: {e}")
                await asyncio.sleep(0.5)
    return StreamingResponse(agen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

# ───────────────────────────────────────────────────────────────────────────────
# 가벼운 테스트 스트림 (CPU 부담 낮춤)
# ───────────────────────────────────────────────────────────────────────────────
@app.get("/test_stream")
async def test_stream():
    boundary = "frame"
    H, W = 480, 640
    xs = np.linspace(0, 1, W, dtype=np.float32)
    ys = np.linspace(0, 1, H, dtype=np.float32)
    grad_x = (xs * 255).astype(np.uint8)[None, :].repeat(H, axis=0)
    grad_y = (ys * 255).astype(np.uint8)[:, None].repeat(W, axis=1)
    async def agen():
        t = 0.0
        while True:
            try:
                b = grad_x
                g = grad_y
                r_val = int((0.5 + 0.5*np.sin(t)) * 255)
                r = np.full_like(b, r_val)
                img = np.dstack([b, g, r])
                cv2.putText(img, "TEST STREAM", (200, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                cv2.putText(img, time.strftime("Time: %H:%M:%S"), (200, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cx = int(320 + 200*np.sin(t))
                cy = int(240 + 100*np.cos(t))
                cv2.circle(img, (cx, cy), 30, (255,255,0), -1)
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                yield (b"--" + boundary.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
                t += 0.1
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"[test_stream] error: {e}")
                await asyncio.sleep(0.5)
    return StreamingResponse(agen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")   

# ───────────────────────────────────────────────────────────────────────────────
# 단독 실행 시 (개별 프로세스 실행용)
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
