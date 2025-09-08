from typing import Optional, Dict, Any, List, Tuple
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, JSONResponse
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

app = FastAPI(title="Multi-Camera Web Server", version="1.3.0")

# ───────────────────────────────────────────────────────────────────────────────
# 전역 핸들 (main.py에서 주입)
# ───────────────────────────────────────────────────────────────────────────────
webviz: Optional[WebPlanViz] = None
_cam_streams: Dict[str, Any] = {}  # {"Camera 1": stream_obj, ...}
PLAN_IMG = "/workspace/assets/250904_homograph_coordinate-plane2.jpg"

# 바운딩박스/라벨 오버레이 저장소
_cam_overlays: Dict[str, List[Dict[str, Any]]] = {}  # {"Camera 1":[{...}], ...}
_cam_ov_lock = threading.Lock()

# 기본 카메라 이미지 스케일 (원본의 n%)
CAM_IMG_DEFAULT_SCALE = 0.6

# 프레임 추출 시도 순서 (속성명)
_JPEG_ATTRS = ("_jpg", "jpg", "last_jpeg", "jpeg", "jpeg_bytes")
_BGR_ATTRS  = ("bgr", "frame", "last_frame", "image", "_frame")

# ───────────────────────────────────────────────────────────────────────────────
# 주입 API
# ───────────────────────────────────────────────────────────────────────────────
def set_webviz(viz: WebPlanViz):
    global webviz
    webviz = viz
    print("✅ WebPlanViz가 웹 서버에 연결되었습니다")

def set_cam_streams(streams: Dict[str, Any]):
    global _cam_streams
    _cam_streams = streams or {}
    print(f"✅ {len(_cam_streams)}개 카메라 스트림이 웹 서버에 연결되었습니다")

def set_cam_overlays(overlays: Dict[str, List[Dict[str, Any]]]):
    global _cam_overlays
    with _cam_ov_lock:
        _cam_overlays = overlays or {}

def _get_cam_overlays(cam_name: str) -> List[Dict[str, Any]]:
    with _cam_ov_lock:
        return list(_cam_overlays.get(cam_name, []))

# ───────────────────────────────────────────────────────────────────────────────
# 유틸
# ───────────────────────────────────────────────────────────────────────────────
def _jpeg_from_ndarray(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""

def _fallback_plan_frame() -> bytes:
    if os.path.exists(PLAN_IMG):
        img = cv2.imread(PLAN_IMG)
        if img is not None:
            return _jpeg_from_ndarray(img)
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(black, "No Plan Image", (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200,200,200), 2)
    return _jpeg_from_ndarray(black)

def _color_for_id(track_id: Optional[int]) -> tuple:
    if track_id is None:
        return (0, 255, 0)
    h = (int(track_id) * 2654435761) & 0xFFFFFFFF
    return (h & 255, (h >> 8) & 255, (h >> 16) & 255)

def _norm_bbox_to_xyxy(bbox, img_w, img_h, it) -> Optional[tuple]:
    if bbox is None:
        return None
    try:
        arr = np.asarray(bbox, dtype=float).reshape(-1)
    except Exception:
        return None
    if arr.size < 4:
        return None

    src_w = it.get("src_w")
    src_h = it.get("src_h")
    if isinstance(src_w, (int, float)) and isinstance(src_h, (int, float)) and src_w > 0 and src_h > 0:
        sx = img_w / float(src_w)
        sy = img_h / float(src_h)
        def scale_fn(x, y, w=None, h=None):
            return (x * sx, y * sy, (w * sx if w is not None else None), (h * sy if h is not None else None))
    else:
        if np.min(arr[:4]) >= 0.0 and np.max(arr[:4]) <= 1.0001:
            def scale_fn(x, y, w=None, h=None):
                return (x * img_w, y * img_h, (w * img_w if w is not None else None), (h * img_h if h is not None else None))
        else:
            def scale_fn(x, y, w=None, h=None):
                return (x, y, w, h)

    x1 = y1 = x2 = y2 = None
    fmt = (it.get("format") or "").lower()

    # xyxy 동의어(tlbr) 허용
    if fmt in ("xyxy", "x1y1x2y2", "tlbr"):
        x1_, y1_, x2_, y2_ = arr[:4]
        x1, y1, _, _ = scale_fn(x1_, y1_)
        x2, y2, _, _ = scale_fn(x2_, y2_)
    elif fmt == "xywh":
        x, y, w, h = arr[:4]
        x, y, w, h = scale_fn(x, y, w, h)
        x1, y1, x2, y2 = x, y, x + w, y + h
    elif fmt in ("cxcywh", "center"):
        cx, cy, w, h = arr[:4]
        cx, cy, w, h = scale_fn(cx, cy, w, h)
        x1, y1, x2, y2 = cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0
    else:
        a, b, c, d = arr[:4]
        if c > a and d > b:
            x1_, y1_, x2_, y2_ = a, b, c, d
            x1, y1, _, _ = scale_fn(x1_, y1_)
            x2, y2, _, _ = scale_fn(x2_, y2_)
        else:
            if c >= 0 and d >= 0:
                x, y, w, h = a, b, c, d
                x, y, w, h = scale_fn(x, y, w, h)
                x1, y1, x2, y2 = x, y, x + w, y + h
            else:
                cx, cy, w, h = a, b, c, d
                cx, cy, w, h = scale_fn(cx, cy, w, h)
                x1, y1, x2, y2 = cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0

    if x2 is None or y2 is None or x2 <= x1 or y2 <= y1:
        return None

    x1 = int(max(0, min(round(x1), img_w - 1)))
    y1 = int(max(0, min(round(y1), img_h - 1)))
    x2 = int(max(0, min(round(x2), img_w - 1)))
    y2 = int(max(0, min(round(y2), img_h - 1)))
    return x1, y1, x2, y2

def _draw_overlays(img_bgr: np.ndarray, items: List[Dict[str, Any]]) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    print(f"[DEBUG] _draw_overlays: image size {W}x{H}, {len(items)} items")
    for i, it in enumerate(items):
        bbox = it.get("bbox")
        xyxy = _norm_bbox_to_xyxy(bbox, W, H, it)
        if xyxy is None:
            print(f"[DEBUG] Item {i}: unusable bbox → {bbox}")
            continue
        x1, y1, x2, y2 = xyxy
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
        cv2.putText(img_bgr, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img_bgr

def _extract_frame_pair(stream_obj: Any) -> Tuple[Optional[np.ndarray], Optional[bytes]]:
    """
    가능한 속성들을 모두 시도하여 최신 프레임을 가져온다.
    반환: (img_bgr, jpg_bytes)
      - 둘 중 하나만 있을 수도 있음 (나머지는 None)
    """
    # 1) JPEG 바이트 속성
    for attr in _JPEG_ATTRS:
        data = getattr(stream_obj, attr, None)
        if isinstance(data, (bytes, bytearray)) and len(data) > 100:
            # 디코드 가능한지 시도
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return img, bytes(data)
            # 못 디코드면 계속 시도
    # 2) BGR ndarray 속성
    for attr in _BGR_ATTRS:
        img = getattr(stream_obj, attr, None)
        if isinstance(img, np.ndarray) and img.ndim >= 2 and img.size > 0:
            return img, None
    return None, None

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
        info = {"has_frame": False, "frame_type": None, "frame_size": 0,
                "is_alive": False, "overlays": ov_counts.get(name, 0)}
        try:
            # 어떤 속성으로 프레임이 있는지 스캔
            img, jpg = _extract_frame_pair(stream)
            if img is not None:
                info["has_frame"] = True
                info["frame_type"] = "bgr"
                info["frame_size"] = int(img.size)
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
        "cam_streams": list(_cam_streams.keys()),
        "cam_stream_count": len(_cam_streams),
        "stream_info": stream_info,
        "overlay_keys": list(_cam_overlays.keys()),
        "timestamp": time.time(),
    }

@app.get("/overlay_keys")
async def overlay_keys():
    with _cam_ov_lock:
        return JSONResponse({
            "stream_keys": list(_cam_streams.keys()),
            "overlay_keys": list(_cam_overlays.keys()),
            "overlay_counts": {k: len(v) for k, v in _cam_overlays.items()},
        })

# ───────────────────────────────────────────────────────────────────────────────
# 메인 페이지
# ───────────────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    if _cam_streams:
        cam_cards = "".join(
            f"""
            <div class="cam-card">
              <div class="cam-label">{name}</div>
              <img class="cam-img" src="/cam/{name}?scale={CAM_IMG_DEFAULT_SCALE}" alt="{name}"/>
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
          <p>• <strong>Connected Streams:</strong> {len(_cam_streams)} cameras</p>
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
                    frame = _fallback_plan_frame()
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
                stream = _cam_streams.get(cam_name)
                if stream is None:
                    # 스트림 없음 → 대기 이미지
                    wait_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(wait_img, f"No stream: {cam_name}", (50, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    out_img = wait_img
                    if scale != 1.0:
                        h, w = out_img.shape[:2]
                        out_img = cv2.resize(out_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                    out_bytes = _jpeg_from_ndarray(out_img)
                else:
                    img_bgr, jpg_bytes = _extract_frame_pair(stream)

                    # 프레임 확보 실패 시 안내 화면
                    if img_bgr is None and jpg_bytes is None:
                        wait_img = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(wait_img, f"Waiting stream: {cam_name}", (50, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        out_img = wait_img
                        if scale != 1.0:
                            h, w = out_img.shape[:2]
                            out_img = cv2.resize(out_img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
                        out_bytes = _jpeg_from_ndarray(out_img)

                    else:
                        # 원본 이미지 확보 (필요 시 JPEG → BGR 디코드)
                        if img_bgr is None and jpg_bytes is not None:
                            nparr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        overlays = _get_cam_overlays(cam_name)
                        if overlays:
                            print(f"[DEBUG] {cam_name}: {len(overlays)} overlays found")
                            img_bgr = _draw_overlays(img_bgr, overlays)

                        # 스케일 다운
                        if scale != 1.0:
                            h, w = img_bgr.shape[:2]
                            img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

                        out_bytes = _jpeg_from_ndarray(img_bgr)

                yield (b"--" + boundary.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + out_bytes + b"\r\n")
                await asyncio.sleep(0.05)
            except Exception as e:
                print(f"[stream_cam:{cam_name}] error: {e}")
                await asyncio.sleep(0.5)
    return StreamingResponse(agen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

# ───────────────────────────────────────────────────────────────────────────────
# 가벼운 테스트 스트림
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
# 단독 실행
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
