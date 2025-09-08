# /workspace/tools/web/app.py
"""
멀티카메라 웹 애플리케이션
- 추적 시스템과 웹 서버 연동
- 추론 루프 실행
- 오버레이 데이터 생성
"""
import logging
import time
import asyncio
import sys
from typing import List, Optional, Dict, Any

sys.path.append("/workspace")

# MCMT는 지연 임포트로 처리 (순환 임포트 방지)
from .manager import WebServerManager
from .server import set_webviz, set_cam_streams, set_cam_overlays, set_class_map

class MultiCameraWebApp:
    """멀티카메라 웹 애플리케이션"""
    def __init__(self, detector_models: Optional[List[str]] = None):
        # 지연 임포트로 순환 임포트 방지
        from MCMT import MultiCameraTrackingSystem
        self.tracking_system: Optional[MultiCameraTrackingSystem] = None
        self.web_manager = WebServerManager()
        self.detector_models = detector_models

    async def run(self):
        print("🚀 멀티카메라 추적 웹 시스템 시작")
        try:
            print("\n=== 1단계: 추적 시스템 초기화 ===")
            if self.detector_models:
                print(f"🎯 사용할 탐지 모델: {self.detector_models}")
            # 지연 임포트로 순환 임포트 방지
            from MCMT import create_tracking_system
            self.tracking_system = create_tracking_system(detector_models=self.detector_models)
            print("✅ 추적 시스템 초기화 완료")

            print("\n=== 2단계: 웹 서버 시작 ===")
            if self.web_manager.start_web_server():
                if getattr(self.tracking_system, "viz", None):
                    set_webviz(self.tracking_system.viz)
                if getattr(self.tracking_system, "streams", None):
                    set_cam_streams(self.tracking_system.streams)
                self._inject_class_map()
                self.web_manager.open_browser()
            else:
                print("❌ 웹 서버 시작 실패 - 웹 인터페이스 비활성화")

            print("\n=== 3단계: 추론 엔진 실행 ===")
            await self._run_inference_loop()

        except KeyboardInterrupt:
            print("\n⏹️ 사용자에 의해 중지됨")
        except Exception as e:
            print(f"❌ 시스템 오류: {e}")
        finally:
            self.cleanup()

    def _inject_class_map(self):
        ts = self.tracking_system
        if ts is None:
            return
        candidates = []
        for attr in ("detector", "detection_api"):
            candidates.append(getattr(ts, attr, None))
        eng = getattr(ts, "engine", None)
        if eng is not None:
            for attr in ("detector", "detection_api", "core"):
                candidates.append(getattr(eng, attr, None))
            core = getattr(eng, "core", None)
            if core is not None:
                candidates.append(getattr(core, "detector", None))
        seen = set()
        for obj in candidates:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            if hasattr(obj, "name_map"):
                try:
                    cmap = obj.name_map()
                    if isinstance(cmap, dict) and cmap:
                        set_class_map(cmap)
                        print(f"✅ class_map injected into web ({len(cmap)} classes)")
                        return
                except Exception as e:
                    print(f"[WARN] name_map() fetch failed: {e}")

    async def _run_inference_loop(self):
        if not self.tracking_system or not self.tracking_system.engine:
            print("❌ 추론 엔진이 초기화되지 않았습니다")
            return

        print("🔄 추론 루프 시작...")
        print(f"[DEBUG] tracking_system.engine = {self.tracking_system.engine}")
        print(f"[DEBUG] tracking_system.streams = {getattr(self.tracking_system, 'streams', None)}")

        last_seen = {"t": time.time(), "round": -1}
        loop_count = 0

        async def watchdog(timeout_sec: int, stall_evt: asyncio.Event):
            while True:
                await asyncio.sleep(2)
                if time.time() - last_seen["t"] > timeout_sec:
                    logging.error("STALL detected → recycling all")
                    stall_evt.set()

        while True:
            loop_count += 1
            print(f"[DEBUG] === 추론 루프 시작 (loop #{loop_count}) ===")
            stall_evt = asyncio.Event()
            wd_task = asyncio.create_task(watchdog(20, stall_evt))

            try:
                print(f"[DEBUG] engine.stream() 호출 시작...")
                async for result in self.tracking_system.engine.stream():
                    if stall_evt.is_set():
                        print(f"[DEBUG] 스톨 이벤트 감지됨, 루프 종료")
                        break

                    last_seen["t"] = time.time()
                    last_seen["round"] = result.get("round", -1)

                    print(f"[DEBUG] 추론 결과 수신: round={result.get('round', -1)}, cameras={len(result.get('cameras', []))}")
                    print(f"[DEBUG] result keys: {list(result.keys())}")

                    print(f"[DEBUG] _create_overlays 호출 시작...")
                    overlays = self._create_overlays(result)
                    print(f"[DEBUG] _create_overlays 완료: {len(overlays)} 카메라")
                    print(f"[DEBUG] set_cam_overlays 호출...")
                    set_cam_overlays(overlays)
                    print(f"[DEBUG] set_cam_overlays 완료")

                    all_coords: List[Any] = []
                    coords_per_cam: List[List[Any]] = []
                    for cam in result.get("cameras", []):
                        pts = cam.get("plan_coords", []) or []
                        all_coords.extend(pts)
                        coords_per_cam.append(pts)

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
        """추론 결과에서 오버레이 데이터 생성 (TrackingAPI 결과 + Detection ndarray/tensor 안전 처리)"""
        import numpy as np
        try:
            import torch
        except Exception:
            torch = None

        overlays: Dict[str, List[Dict[str, Any]]] = {}

        try:
            stream_names = list(self.tracking_system.streams.keys()) if self.tracking_system and self.tracking_system.streams else []
        except Exception:
            stream_names = []

        print(f"[DEBUG] _create_overlays: processing {len(result.get('cameras') or [])} cameras")
        print(f"[DEBUG] _create_overlays: stream_names = {stream_names}")

        def _append_src_wh(item: Dict[str, Any], fw: Optional[int], fh: Optional[int]):
            if fw and fh:
                item["src_w"] = float(fw)
                item["src_h"] = float(fh)

        def _tlwh_or_xyxy_to_xyxy(b: np.ndarray) -> tuple:
            x, y, w, h = b[:4]
            if (w > 0 and h > 0) and not (b[2] > b[0] and b[3] > b[1]):
                return float(x), float(y), float(x + w), float(y + h)
            else:
                return float(b[0]), float(b[1]), float(b[2]), float(b[3])

        for idx, cam in enumerate(result.get("cameras") or []):
            items: List[Dict[str, Any]] = []
            cam_name = stream_names[idx] if idx < len(stream_names) else f"cam{idx+1}"
            print(f"[DEBUG] {cam_name}: 카메라 처리 시작")

            fh = fw = None
            shape = cam.get("frame_shape", None)
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                fh, fw = int(shape[0]), int(shape[1])
            print(f"[DEBUG] {cam_name}: frame_shape={shape}")

            # Tracklets: list[dict]
            tracklets_raw = cam.get("tracklets", None)
            if isinstance(tracklets_raw, list):
                print(f"[DEBUG] {cam_name}: {len(tracklets_raw)} tracklets, {cam.get('detection_count', '?')} detections")
                for i, t in enumerate(tracklets_raw):
                    try:
                        bbox = t.get("bbox", None)
                        if bbox is None:
                            continue
                        b = np.asarray(bbox, dtype=float).reshape(-1)
                        if b.size < 4:
                            continue
                        x1, y1, x2, y2 = _tlwh_or_xyxy_to_xyxy(b)
                        item = {
                            "bbox": [x1, y1, x2, y2],
                            "label": t.get("label") or t.get("cls_name") or "unknown",
                            "score": float(t.get("score", 1.0)),
                            "track_id": t.get("id"),
                            "class_id": int(t.get("class_id", -1)) if t.get("class_id") is not None else -1,
                            "format": "xyxy",
                        }
                        _append_src_wh(item, fw, fh)
                        items.append(item)
                    except Exception as e:
                        print(f"[DEBUG] {cam_name}: tracklet {i} error: {e}")
            elif tracklets_raw is not None:
                print(f"[DEBUG] {cam_name}: tracklets 타입 비정상 -> {type(tracklets_raw)} (무시)")

            # Detections fallback
            if not items:
                det_raw = cam.get("detections", None)
                print(f"[DEBUG] {cam_name}: tracklets 없음 → detections 사용 시도, 타입={type(det_raw)}")

                det_arr = None
                if isinstance(det_raw, np.ndarray):
                    det_arr = det_raw
                elif torch is not None and isinstance(det_raw, torch.Tensor):
                    det_arr = det_raw.detach().cpu().numpy()

                if det_arr is not None:
                    if det_arr.ndim == 2 and det_arr.shape[1] >= 6 and det_arr.size > 0:
                        N = det_arr.shape[0]
                        print(f"[DEBUG] {cam_name}: ndarray/tensor detections {N}개 변환")
                        for i in range(N):
                            try:
                                x1, y1, x2, y2, score, cid = det_arr[i, 0:6].tolist()
                                item = {
                                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                    "label": "unknown",
                                    "score": float(score),
                                    "track_id": None,
                                    "class_id": int(cid),
                                    "format": "xyxy",
                                }
                                _append_src_wh(item, fw, fh)
                                items.append(item)
                            except Exception as e:
                                print(f"[DEBUG] {cam_name}: detection(np) {i} error: {e}")
                    else:
                        print(f"[DEBUG] {cam_name}: det_arr shape 불가: {getattr(det_arr, 'shape', None)}")

                elif isinstance(det_raw, list) and det_raw:
                    all_dicts = all(isinstance(x, dict) for x in det_raw)
                    if all_dicts:
                        for i, d in enumerate(det_raw):
                            try:
                                bbox = d.get("bbox")
                                if bbox is None:
                                    continue
                                b = np.asarray(bbox, dtype=float).reshape(-1)
                                if b.size < 4:
                                    continue
                                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                                item = {
                                    "bbox": [x1, y1, x2, y2],
                                    "label": d.get("label") or d.get("cls_name") or "unknown",
                                    "score": float(d.get("score", 1.0)),
                                    "track_id": None,
                                    "class_id": int(d.get("class_id", -1)) if d.get("class_id") is not None else -1,
                                    "format": "xyxy",
                                }
                                _append_src_wh(item, fw, fh)
                                items.append(item)
                            except Exception as e:
                                print(f"[DEBUG] {cam_name}: detection(dict) {i} error: {e}")
                    else:
                        try:
                            det_arr = np.asarray(det_raw, dtype=float)
                        except Exception:
                            det_arr = None
                        if det_arr is not None and det_arr.ndim == 2 and det_arr.shape[1] >= 6:
                            N = det_arr.shape[0]
                            for i in range(N):
                                try:
                                    x1, y1, x2, y2, score, cid = det_arr[i, 0:6].tolist()
                                    item = {
                                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                        "label": "unknown",
                                        "score": float(score),
                                        "track_id": None,
                                        "class_id": int(cid),
                                        "format": "xyxy",
                                    }
                                    _append_src_wh(item, fw, fh)
                                    items.append(item)
                                except Exception as e:
                                    print(f"[DEBUG] {cam_name}: detection(list->np) {i} error: {e}")

            overlays[cam_name] = items
            print(f"[DEBUG] {cam_name}: final items = {len(items)}")

        print(f"[DEBUG] overlay_keys={list(overlays.keys())}, stream_keys={stream_names}")
        return overlays

    def cleanup(self):
        print("🧹 리소스 정리 중...")
        self.web_manager.stop_web_server()
        if self.tracking_system:
            try:
                self.tracking_system.cleanup()
            except Exception:
                pass
        print("✅ 정리 완료")
