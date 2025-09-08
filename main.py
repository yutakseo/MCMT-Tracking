# /workspace/main.py - Multi-Camera Tracking Web Server
"""
멀티카메라 추적 시스템 웹 서버 실행
- MCMT.py의 핵심 시스템과 연동
- 웹 서버 자동 실행 및 관리
- 브라우저 자동 열기
"""

import logging
import time
import asyncio
import sys
import threading
import webbrowser
import socket
import urllib.request
from typing import List, Optional, Dict, Any

import numpy as np

sys.path.append("/workspace")

# MCMT 핵심 시스템과 웹 앱 import
from MCMT import create_tracking_system, MultiCameraTrackingSystem
from app_web import app, set_webviz, set_cam_streams, set_cam_overlays

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")


# =============================================================================
# 유틸: 배열/텐서의 진리값 평가를 피하기 위한 안전 래퍼
# =============================================================================
def _safe_seq(x) -> List[Any]:
    """x가 None/Tensor/ndarray/iterable 어느 것이든 '불리언 평가 없이' 리스트로 돌려준다."""
    if x is None:
        return []
    # 이미 리스트/튜플이면 즉시 복사
    if isinstance(x, (list, tuple)):
        return list(x)
    # numpy/torch는 tolist 우선 시도
    try:
        if hasattr(x, "tolist"):
            return x.tolist()
    except Exception:
        pass
    # 마지막 시도: iterable로 캐스팅
    try:
        return list(x)
    except Exception:
        return []

def _safe_points(x) -> List[Any]:
    """plan_coords 처럼 포인트 묶음을 안전 변환."""
    pts = _safe_seq(x)
    # 각 원소도 2원소로 강제 변환(가능한 경우)
    out = []
    for p in pts:
        try:
            if isinstance(p, (list, tuple, np.ndarray)) and len(p) >= 2:
                out.append((float(p[0]), float(p[1])))
        except Exception:
            continue
    return out


# =============================================================================
# 웹 서버 관리자
# =============================================================================
class WebServerManager:
    """웹 서버 자동 관리"""

    def __init__(self):
        self.web_thread = None
        self.web_port = 8000
        self.web_url = f"http://localhost:{self.web_port}"

    def _port_in_use(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            return s.connect_ex(("127.0.0.1", port)) == 0

    def _probe_healthz(self, url: str) -> bool:
        try:
            with urllib.request.urlopen(url, timeout=0.6) as r:
                return r.read(2) == b"ok"
        except Exception:
            return False

    def _find_free_port(self, start=8000, end=8100) -> int:
        for p in range(start, end + 1):
            if not self._port_in_use(p):
                return p
        raise RuntimeError("No free port found in range 8000-8100")

    def start_web_server(self) -> bool:
        """웹 서버 시작"""
        try:
            # 기존 서버 재사용 시도
            if self._port_in_use(self.web_port):
                if self._probe_healthz(f"http://127.0.0.1:{self.web_port}/healthz"):
                    print(f"♻️ 기존 웹 서버 재사용: {self.web_url}")
                    return True
                else:
                    # 헬스체크 실패 → 새 포트 찾기
                    new_port = self._find_free_port()
                    print(f"⚠️ 포트 {self.web_port} 사용중(헬스 실패). 포트를 {new_port}로 변경합니다.")
                    self.web_port = new_port
                    self.web_url = f"http://localhost:{self.web_port}"

            print("🌐 웹 서버 시작 중...")

            def _run():
                import uvicorn
                uvicorn.run(app, host="0.0.0.0", port=self.web_port, log_level="info")

            self.web_thread = threading.Thread(target=_run, daemon=True)
            self.web_thread.start()

            # 부팅 대기 및 헬스체크
            for _ in range(40):
                if self._probe_healthz(f"http://127.0.0.1:{self.web_port}/healthz"):
                    print(f"✅ 웹 서버 시작 완료: {self.web_url}")
                    return True
                time.sleep(0.2)

            print("❌ 웹 서버 시작 확인 실패(헬스체크 타임아웃)")
            return False

        except Exception as e:
            print(f"❌ 웹 서버 시작 오류: {e}")
            return False

    def open_browser(self):
        """웹 브라우저 자동 열기"""
        try:
            print("🌐 웹 브라우저 열기...")
            webbrowser.open(self.web_url)
            print(f"✅ 브라우저에서 {self.web_url} 열림")
        except Exception as e:
            print(f"❌ 브라우저 열기 실패: {e}")
            print(f"수동으로 {self.web_url}에 접속하세요")

    def stop_web_server(self):
        """웹 서버 종료"""
        print("ℹ️ uvicorn 스레드는 프로세스 종료 시 함께 정리됩니다.")


# =============================================================================
# 메인 애플리케이션
# =============================================================================
class MultiCameraWebApp:
    """멀티카메라 웹 애플리케이션"""

    def __init__(self, detector_models: Optional[List[str]] = None):
        self.tracking_system: Optional[MultiCameraTrackingSystem] = None
        self.web_manager = WebServerManager()
        self.detector_models = detector_models

    async def run(self):
        """메인 실행 루프"""
        print("🚀 멀티카메라 추적 웹 시스템 시작")

        try:
            # 1단계: 추적 시스템 초기화
            print("\n=== 1단계: 추적 시스템 초기화 ===")
            if self.detector_models:
                print(f"🎯 사용할 탐지 모델: {self.detector_models}")
            self.tracking_system = create_tracking_system(detector_models=self.detector_models)
            print("✅ 추적 시스템 초기화 완료")

            # 2단계: 웹 서버 시작
            print("\n=== 2단계: 웹 서버 시작 ===")
            if self.web_manager.start_web_server():
                # 웹 시각화와 스트림을 웹 서버에 연결
                if getattr(self.tracking_system, "viz", None):
                    set_webviz(self.tracking_system.viz)
                if getattr(self.tracking_system, "streams", None):
                    set_cam_streams(self.tracking_system.streams)

                # 브라우저 자동 열기
                self.web_manager.open_browser()
            else:
                print("❌ 웹 서버 시작 실패 - 웹 인터페이스 비활성화")

            # 3단계: 추론 엔진 실행
            print("\n=== 3단계: 추론 엔진 실행 ===")
            await self._run_inference_loop()

        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중지됨")
        except Exception as e:
            print(f"❌ 시스템 오류: {e}")
        finally:
            self.cleanup()

    async def _run_inference_loop(self):
        """추론 루프 실행"""
        if not self.tracking_system or not self.tracking_system.engine:
            print("❌ 추론 엔진이 초기화되지 않았습니다")
            return

        print("🔄 추론 루프 시작...")

        # 스톨 감지용 변수
        last_seen = {"t": time.time(), "round": -1}

        async def watchdog(timeout_sec: int, stall_evt: asyncio.Event):
            while True:
                await asyncio.sleep(2)
                if time.time() - last_seen["t"] > timeout_sec:
                    logging.error("STALL detected → recycling all")
                    stall_evt.set()

        # 메인 루프
        while True:
            stall_evt = asyncio.Event()
            wd_task = asyncio.create_task(watchdog(20, stall_evt))

            try:
                async for result in self.tracking_system.engine.stream():
                    if stall_evt.is_set():
                        break

                    last_seen["t"] = time.time()
                    last_seen["round"] = result.get("round", -1)

                    # 오버레이 데이터 생성 및 전달
                    overlays = self._create_overlays(result)
                    logging.debug(f"[OVERLAY] counts: {{ {', '.join(f'{k}: {len(v)}' for k, v in overlays.items())} }}")

                    # 스트림 키와 오버레이 키가 일치하는지 점검
                    try:
                        stream_names = list(self.tracking_system.streams.keys())
                        missing = [k for k in overlays.keys() if k not in stream_names]
                        if missing:
                            logging.warning(f"[OVERLAY] keys not in streams: {missing} | streams={stream_names}")
                    except Exception:
                        pass

                    set_cam_overlays(overlays)

                    # 좌표 취합 (불리언 평가 없이 안전 처리)
                    all_coords: List[Any] = []
                    coords_per_cam: List[List[Any]] = []
                    for cam in _safe_seq(result.get("cameras")):
                        pts = _safe_points(cam.get("plan_coords"))
                        all_coords.extend(pts)
                        coords_per_cam.append(pts)

                    # 웹 시각화 업데이트
                    if getattr(self.tracking_system, "viz", None):
                        try:
                            self.tracking_system.viz.update({
                                "round": result.get("round"),
                                "timestamp": result.get("timestamp"),
                                "fused": all_coords,
                                "coords": coords_per_cam,
                            })
                        except Exception as e:
                            logging.error(f"[VIZ] update failed: {e}")

                    # 로깅
                    gpu_state = result.get("gpu_state", "UNKNOWN")
                    gpu_util = result.get("gpu_utilization", 0.0)
                    batch_time = result.get("batch_time", 0.0)
                    total_detections = result.get("total_detections", 0)
                    total_tracks = result.get("total_tracks", 0)

                    logging.info(
                        f"[SYSTEM] round={result.get('round', -1)} objects={len(all_coords)} "
                        f"gpu={gpu_state}({gpu_util:.1f}%) batch={batch_time:.3f}s "
                        f"det={total_detections} track={total_tracks}"
                    )

            except Exception as e:
                logging.error(f"[SYSTEM] recycling after error: {e}")
            finally:
                # 정리
                try:
                    self.tracking_system.engine.stop()
                except Exception:
                    pass

                try:
                    wd_task.cancel()
                except Exception:
                    pass

                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                await asyncio.sleep(2)

    def _create_overlays(self, result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """추론 결과에서 오버레이 데이터 생성 (스트림 키와 이름을 정확히 매칭)"""
        overlays: Dict[str, List[Dict[str, Any]]] = {}

        # 현재 웹서버가 알고 있는 실제 스트림 키들 (index 순서 중요)
        try:
            stream_names = list(self.tracking_system.streams.keys()) if self.tracking_system and self.tracking_system.streams else []
        except Exception:
            stream_names = []

        print(f"[DEBUG] _create_overlays: processing {len(_safe_seq(result.get('cameras')))} cameras")
        print(f"[DEBUG] _create_overlays: stream_names = {stream_names}")

        for idx, cam in enumerate(_safe_seq(result.get("cameras"))):
            items: List[Dict[str, Any]] = []

            # 1) 이름 매핑: cam dict에 name이 있으면 우선 사용, 없으면 streams 순서와 매칭, 최후엔 cam{idx+1}
            cam_name = cam.get("name") or (stream_names[idx] if idx < len(stream_names) else f"cam{idx+1}")

            # 2) 원본 프레임 크기
            fh, fw = None, None
            shape = cam.get("frame_shape")
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                fh, fw = int(shape[0]), int(shape[1])
            print(f"[DEBUG] {cam_name}: frame_shape={shape}")

            # 3) 소스들 (불리언 평가 없이 안전 변환)
            tracklets = _safe_seq(cam.get("tracklets"))
            detections = _safe_seq(cam.get("detections"))
            print(f"[DEBUG] {cam_name}: {len(tracklets)} tracklets, {len(detections)} detections")

            # 4) tracklets → tlwh 가정하여 xyxy로 변환(이미 xyxy일 수 있어 휴리스틱)
            for i, t in enumerate(tracklets):
                try:
                    if not isinstance(t, dict):
                        # dict가 아니면 스킵 (포맷 미상)
                        continue
                    bbox = t.get("bbox")
                    if bbox is None:
                        continue

                    # ndarray/텐서 안전 변환
                    b = np.asarray(bbox, dtype=float).reshape(-1)
                    if b.size < 4:
                        continue

                    # tlwh -> xyxy 또는 이미 xyxy
                    x, y, w, h = b[:4]
                    if (w > 0 and h > 0) and not (b[2] > b[0] and b[3] > b[1]):
                        x1, y1, x2, y2 = x, y, x + w, y + h
                    else:
                        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]

                    label = t.get("label") or t.get("cls_name") or "unknown"
                    class_id = int(t.get("class_id", -1)) if t.get("class_id") is not None else -1
                    if class_id != -1:
                        label = f"{label}({class_id})"

                    item: Dict[str, Any] = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "label": label,
                        "score": float(t.get("score", 1.0)),
                        "track_id": t.get("track_id") or t.get("id"),
                        "class_id": class_id,
                        "format": "xyxy",  # 명시적으로 xyxy로 표시
                    }
                    if fw and fh:
                        item["src_w"] = float(fw)
                        item["src_h"] = float(fh)
                    items.append(item)
                    # 상세 로그는 과다해질 수 있어 필요 시 주석 해제
                    # print(f"[DEBUG] {cam_name}: tracklet {i} -> {item}")
                except Exception as e:
                    print(f"[DEBUG] {cam_name}: tracklet {i} error: {e}")

            # 5) tracklets가 없으면 detections 사용 (DetectionAPI는 보통 xyxy)
            if len(items) == 0:
                for i, d in enumerate(detections):
                    try:
                        if not isinstance(d, dict):
                            continue
                        bbox = d.get("bbox")
                        if bbox is None:
                            continue

                        b = np.asarray(bbox, dtype=float).reshape(-1)
                        if b.size < 4:
                            continue

                        label = d.get("label") or d.get("cls_name") or "unknown"
                        class_id = int(d.get("class_id", -1)) if d.get("class_id") is not None else -1
                        if class_id != -1:
                            label = f"{label}({class_id})"

                        item: Dict[str, Any] = {
                            "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                            "label": label,
                            "score": float(d.get("score", 1.0)),
                            "track_id": None,
                            "class_id": class_id,
                            "format": "xyxy",  # 명시
                        }
                        if fw and fh:
                            item["src_w"] = float(fw)
                            item["src_h"] = float(fh)
                        items.append(item)
                    except Exception as e:
                        print(f"[DEBUG] {cam_name}: detection {i} error: {e}")

            overlays[cam_name] = items
            print(f"[DEBUG] {cam_name}: final items = {len(items)}")

        return overlays

    def cleanup(self):
        """리소스 정리"""
        print("🧹 리소스 정리 중...")

        # 웹 서버 종료
        self.web_manager.stop_web_server()

        # 추적 시스템 정리
        if self.tracking_system:
            try:
                self.tracking_system.cleanup()
            except Exception:
                pass

        print("✅ 정리 완료")


# =============================================================================
# 메인 실행
# =============================================================================
async def main(detector_models: Optional[List[str]] = None):
    """메인 함수

    Args:
        detector_models: 사용할 탐지 모델 리스트 (예: ["vehicle"], ["ultra_people"], ["vehicle", "ultra_people"])
    """
    app = MultiCameraWebApp(detector_models=detector_models)
    await app.run()


if __name__ == "__main__":
    # 3. 사람+작업자+차량
    asyncio.run(main(detector_models=["ultra_people", "worker", "vehicle"]))
