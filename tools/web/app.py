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

from MCMT import create_tracking_system, MultiCameraTrackingSystem
from .manager import WebServerManager
from .server import set_webviz, set_cam_streams, set_cam_overlays, set_class_map


class MultiCameraWebApp:
    """멀티카메라 웹 애플리케이션"""

    def __init__(self, detector_models: Optional[List[str]] = None):
        self.tracking_system: Optional[MultiCameraTrackingSystem] = None
        self.web_manager = WebServerManager()
        self.detector_models = detector_models

    async def run(self):
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
                if getattr(self.tracking_system, "viz", None):
                    set_webviz(self.tracking_system.viz)
                if getattr(self.tracking_system, "streams", None):
                    set_cam_streams(self.tracking_system.streams)

                # DetectionAPI의 class_map을 웹에 1회 주입
                self._inject_class_map()

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

    def _inject_class_map(self):
        """
        DetectionAPI(또는 동등 객체)의 name_map()을 찾아 tools.web에 주입.
        트리: tracking_system, engine, core, detector 등 다양한 곳을 안전 탐색.
        """
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
        cmap: Dict[int, str] = {}
        for obj in candidates:
            if obj is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            if hasattr(obj, "name_map"):
                try:
                    maybe = obj.name_map()
                    if isinstance(maybe, dict) and maybe:
                        cmap = {int(k): str(v) for k, v in maybe.items()}
                        break
                except Exception as e:
                    print(f"[WARN] name_map() fetch failed: {e}")

        # ✅ 빈 맵이면 기본 맵으로 폴백
        if not cmap:
            try:
                from __Detection.detection_api import DEFAULT_CLASS_MAP
                cmap = {int(k): str(v) for k, v in DEFAULT_CLASS_MAP.items()}
                print(f"[WARN] class_map not found from engine → fallback to DEFAULT_CLASS_MAP ({len(cmap)} classes)")
            except Exception:
                cmap = {}

        set_class_map(cmap)
        print(f"✅ class_map injected into web ({len(cmap)} classes)")

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

                    # 오버레이 데이터 생성 및 전달
                    print(f"[DEBUG] _create_overlays 호출 시작...")
                    overlays = self._create_overlays(result)
                    print(f"[DEBUG] _create_overlays 완료: {len(overlays)} 카메라")
                    print(f"[DEBUG] set_cam_overlays 호출...")
                    set_cam_overlays(overlays)
                    print(f"[DEBUG] set_cam_overlays 완료")

                    # 좌표 취합
                    all_coords: List[Any] = []
                    coords_per_cam: List[List[Any]] = []
                    for cam in result.get("cameras", []):
                        pts = cam.get("plan_coords", []) or []
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

        # /cam/{key}와 1:1 매칭되는 스트림 키
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

        def _to_xyxy(b: np.ndarray, fmt: Optional[str] = None) -> tuple:
            """
            bbox를 xyxy로 변환.
            - fmt 힌트가 있으면 우선 적용: 'xyxy' | 'xywh'/'tlwh' | 'cxcywh'
            - 없으면 (c>a and d>b)이면 xyxy, (c>=0 and d>=0)이면 tlwh, 아니면 cxcywh 가정
            """
            a, b1, c, d = b[:4]
            fmt = (fmt or "").lower()

            if fmt in ("xyxy", "x1y1x2y2", "tlbr"):
                return float(a), float(b1), float(c), float(d)
            elif fmt in ("xywh", "tlwh"):
                x, y, w, h = float(a), float(b1), float(c), float(d)
                return x, y, x + w, y + h
            elif fmt in ("cxcywh", "center"):
                cx, cy, w, h = float(a), float(b1), float(c), float(d)
                return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0

            # 휴리스틱
            if c > a and d > b1:
                # xyxy로 간주
                return float(a), float(b1), float(c), float(d)
            else:
                if c >= 0 and d >= 0:
                    # tlwh
                    x, y, w, h = float(a), float(b1), float(c), float(d)
                    return x, y, x + w, y + h
                else:
                    # cxcywh
                    cx, cy, w, h = float(a), float(b1), float(c), float(d)
                    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0

        for idx, cam in enumerate(result.get("cameras") or []):
            items: List[Dict[str, Any]] = []

            cam_name = stream_names[idx] if idx < len(stream_names) else f"cam{idx+1}"
            print(f"[DEBUG] {cam_name}: 카메라 처리 시작")

            # 프레임 크기
            fh = fw = None
            shape = cam.get("frame_shape", None)
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                fh, fw = int(shape[0]), int(shape[1])
            print(f"[DEBUG] {cam_name}: frame_shape={shape}")

            # ── TRACKLETS ──────────────────────────────────────────────────────
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
                        fmt_hint = t.get("format")  # 있으면 사용
                        x1, y1, x2, y2 = _to_xyxy(b, fmt_hint)

                        item = {
                            "bbox": [x1, y1, x2, y2],
                            # label은 utils의 숫자문자열 필터에 걸리도록 'unknown' 권장
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

            # ── DETECTIONS (fallback) ──────────────────────────────────────────
            if not items:
                det_raw = cam.get("detections", None)
                print(f"[DEBUG] {cam_name}: tracklets 없음 → detections 사용 시도, 타입={type(det_raw)}")

                det_arr = None
                if isinstance(det_raw, np.ndarray):
                    det_arr = det_raw
                elif torch is not None and isinstance(det_raw, torch.Tensor):
                    det_arr = det_raw.detach().cpu().numpy()

                # case A) ndarray / tensor: (N,6) [x1,y1,x2,y2,score,class_id]
                if det_arr is not None:
                    if det_arr.ndim == 2 and det_arr.shape[1] >= 6 and det_arr.size > 0:
                        N = det_arr.shape[0]
                        print(f"[DEBUG] {cam_name}: ndarray/tensor detections {N}개 변환")
                        for i in range(N):
                            try:
                                x1, y1, x2, y2, score, cid = det_arr[i, 0:6].tolist()
                                item = {
                                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                    "label": "unknown",  # overlay에서 class_map 사용
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

                # case B) list[dict]
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
                                fmt_hint = d.get("format")
                                x1, y1, x2, y2 = _to_xyxy(b, fmt_hint)
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
                        # list지만 dict가 아니면 행렬로 변환 시도
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
