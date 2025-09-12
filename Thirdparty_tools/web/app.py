# /workspace/tools/web/app.py
"""
멀티카메라 웹 애플리케이션
- 추적 시스템과 웹 서버 연동
- 추론 루프 실행
- 오버레이 데이터 생성
- (폴백) 디텍션/트랙 bbox → 플랜 좌표 투영
"""

import logging
import time
import asyncio
import sys
from typing import List, Optional, Dict, Any, Tuple

sys.path.append("/workspace")

from MCMT import create_tracking_system, MultiCameraTrackingSystem
from .manager import WebServerManager
from .server import set_webviz, set_cam_streams, set_cam_overlays, set_class_map


class MultiCameraWebApp:
    """멀티카메라 웹 애플리케이션"""

    def __init__(
        self,
        detector_models: Optional[List[str]] = None,
        draw_mode: str = "auto",
        # 투영 관련 옵션
        project_footpoint: str = "bottom_center",  # "bottom_center" | "center"
        min_score_for_projection: float = 0.05,    # 너무 낮은 박스는 플랜 투영 제외
    ):
        """
        Args:
            detector_models: 사용할 탐지 모델 리스트
            draw_mode:
              - "auto"   : 트랙이 있으면 트랙, 없으면 디텍션(기본)
              - "dets"   : 항상 디텍션으로 그림
              - "tracks" : 항상 트랙만 그림
              - "both"   : 트랙 + 디텍션 모두 그림
            project_footpoint: 플랜 투영에 사용할 점 (bbox의 중심 or 바닥중앙)
            min_score_for_projection: 이 점수 미만인 박스는 플랜 투영에서 제외
        """
        if draw_mode not in ("auto", "dets", "tracks", "both"):
            raise ValueError("draw_mode must be one of: auto, dets, tracks, both")

        self.tracking_system: Optional[MultiCameraTrackingSystem] = None
        self.web_manager = WebServerManager()
        self.detector_models = detector_models
        self.draw_mode = draw_mode

        self.project_footpoint = project_footpoint
        self.min_score_for_projection = float(min_score_for_projection)

    async def run(self):
        print("🚀 멀티카메라 추적 웹 시스템 시작")

        try:
            # 1단계: 추적 시스템 초기화
            print("\n=== 1단계: 추적 시스템 초기화 ===")
            if self.detector_models:
                print(f"🎯 사용할 탐지 모델: {self.detector_models}")
            print(f"🖼️ 오버레이 모드(draw_mode): {self.draw_mode}")
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

                    # 1) 프레임 오버레이 생성
                    overlays = self._create_overlays(result)
                    set_cam_overlays(overlays)

                    # 2) 플랜 좌표 업데이트
                    fused, coords_per_cam = self._collect_or_project_plan_points(result, overlays)
                    if getattr(self.tracking_system, "viz", None):
                        try:
                            self.tracking_system.viz.update({
                                "round": result.get("round"),
                                "timestamp": result.get("timestamp"),
                                "fused": fused,
                                "coords": coords_per_cam,
                            })
                        except Exception as e:
                            logging.error(f"[VIZ] update failed: {e}")

                    # 3) 상태 로깅
                    gpu_state = result.get("gpu_state", "UNKNOWN")
                    gpu_util = result.get("gpu_utilization", 0.0)
                    batch_time = result.get("batch_time", 0.0)
                    total_detections = result.get("total_detections", 0)
                    total_tracks = result.get("total_tracks", 0)
                    logging.info(
                        f"[SYSTEM] round={result.get('round', -1)} "
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

    # ──────────────────────────────────────────────────────────────────────
    # 오버레이(프레임 바운딩박스) 생성
    # ──────────────────────────────────────────────────────────────────────
    def _create_overlays(self, result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """TrackingAPI 결과 + Detection ndarray/tensor 안전 처리 → overlay items"""
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

        want_tracks = self.draw_mode in ("auto", "tracks", "both")
        want_dets   = self.draw_mode in ("auto", "dets", "both")

        # --- helper: bbox가 정의된 '원본 해상도'를 최대한 정확히 추정 ---
        def _pick_src_shape(cam: Dict[str, Any], t: Optional[Dict[str, Any]] = None):
            """
            우선순위:
            (트랙릿/디텍션 항목 내) src_w/src_h, img_w/img_h, frame_w/frame_h
            → (카메라 dict 내) bbox_src_shape, det_src_shape, orig_shape, source_shape, frame_shape
            """
            # item 레벨 우선
            if t:
                for wk, hk in (("src_w","src_h"), ("img_w","img_h"), ("frame_w","frame_h")):
                    w = t.get(wk); h = t.get(hk)
                    if isinstance(w, (int, float)) and isinstance(h, (int, float)) and w > 0 and h > 0:
                        return int(w), int(h)

            # cam 레벨
            for key in ("bbox_src_shape", "det_src_shape", "orig_shape", "source_shape", "frame_shape"):
                shp = cam.get(key)
                if isinstance(shp, (list, tuple)) and len(shp) >= 2:
                    # 일반적으로 (H,W,...) 이므로 0=H,1=W
                    try:
                        fh, fw = int(shp[0]), int(shp[1])
                        if fh > 0 and fw > 0:
                            return fw, fh
                    except Exception:
                        pass

            return None, None  # 알 수 없으면 None

        def _maybe_set_src_wh(item: Dict[str, Any], cam: Dict[str, Any], t: Optional[Dict[str, Any]] = None):
            """
            - bbox가 0~1 범위 정규화처럼 보이면 src_w/h를 넣지 않음(overlay가 자동 스케일)
            - 아니면 _pick_src_shape로 찾은 원본 해상도를 src_w/h에 기록
            """
            bbox = item.get("bbox") or []
            arr = np.asarray(bbox, dtype=float).reshape(-1)
            # 정규화 감지: 0~1.0001 사이
            if arr.size >= 4 and (np.min(arr[:4]) >= 0.0 and np.max(arr[:4]) <= 1.0001):
                return  # 정규화 좌표 → src_w/h 미지정 (overlay가 img 크기에 맞춰 변환)
            fw, fh = _pick_src_shape(cam, t)
            if fw and fh:
                item["src_w"] = float(fw)
                item["src_h"] = float(fh)

        def _to_xyxy(b, fmt: Optional[str] = None):
            arr = np.asarray(b, dtype=float).reshape(-1)
            a, b1, c, d = arr[:4]
            fmt = (fmt or "").lower()
            if fmt in ("xyxy", "x1y1x2y2", "tlbr"):
                return float(a), float(b1), float(c), float(d)
            if fmt in ("xywh", "tlwh"):
                x, y, w, h = float(a), float(b1), float(c), float(d)
                return x, y, x + w, y + h
            if fmt in ("cxcywh", "center"):
                cx, cy, w, h = float(a), float(b1), float(c), float(d)
                return cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0
            # 휴리스틱
            if c > a and d > b1:  # xyxy
                return float(a), float(b1), float(c), float(d)
            if c >= 0 and d >= 0: # tlwh
                x, y, w, h = float(a), float(b1), float(c), float(d)
                return x, y, x + w, y + h
            # center
            cx, cy, w, h = float(a), float(b1), float(c), float(d)
            return cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0

        for idx, cam in enumerate(result.get("cameras") or []):
            items: List[Dict[str, Any]] = []
            cam_name = stream_names[idx] if idx < len(stream_names) else f"cam{idx+1}"

            # TRACKLETS
            if want_tracks:
                tracklets_raw = cam.get("tracklets", None)
                if isinstance(tracklets_raw, list) and tracklets_raw:
                    for t in tracklets_raw:
                        try:
                            bbox = t.get("bbox", None)
                            if bbox is None:
                                continue
                            x1, y1, x2, y2 = _to_xyxy(bbox, t.get("format"))
                            item = {
                                "bbox": [x1, y1, x2, y2],
                                "label": t.get("label") or t.get("cls_name") or "unknown",
                                "score": float(t.get("score", 1.0)),
                                "track_id": t.get("id"),
                                "class_id": int(t.get("class_id", -1)) if t.get("class_id") is not None else -1,
                                "format": "xyxy",
                            }
                            _maybe_set_src_wh(item, cam, t)  # ← 핵심
                            items.append(item)
                        except Exception:
                            continue

            # DETECTIONS (fallback or 추가)
            need_dets = (
                (self.draw_mode == "dets") or
                (self.draw_mode == "both") or
                (self.draw_mode == "auto" and not items)
            )
            if want_dets and need_dets:
                det_raw = cam.get("detections", None)

                # ndarray / tensor: (N,6) [x1,y1,x2,y2,score,class_id]
                det_arr = None
                try:
                    import numpy as _np
                    if isinstance(det_raw, _np.ndarray):
                        det_arr = det_raw
                    elif torch is not None:
                        import torch as _tc
                        if isinstance(det_raw, _tc.Tensor):
                            det_arr = det_raw.detach().cpu().numpy()
                except Exception:
                    det_arr = None

                if det_arr is not None and det_arr.ndim == 2 and det_arr.shape[1] >= 6 and det_arr.size > 0:
                    for i in range(det_arr.shape[0]):
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
                            _maybe_set_src_wh(item, cam, None)  # ← 핵심
                            items.append(item)
                        except Exception:
                            continue
                elif isinstance(det_raw, list) and det_raw:
                    all_dicts = all(isinstance(x, dict) for x in det_raw)
                    if all_dicts:
                        for d in det_raw:
                            try:
                                bbox = d.get("bbox")
                                if bbox is None:
                                    continue
                                x1, y1, x2, y2 = _to_xyxy(bbox, d.get("format"))
                                item = {
                                    "bbox": [x1, y1, x2, y2],
                                    "label": d.get("label") or d.get("cls_name") or "unknown",
                                    "score": float(d.get("score", 1.0)),
                                    "track_id": None,
                                    "class_id": int(d.get("class_id", -1)) if d.get("class_id") is not None else -1,
                                    "format": "xyxy",
                                }
                                _maybe_set_src_wh(item, cam, d)  # ← 핵심
                                items.append(item)
                            except Exception:
                                continue
                    else:
                        # list -> array 시도
                        try:
                            import numpy as _np
                            det_arr = _np.asarray(det_raw, dtype=float)
                            if det_arr.ndim == 2 and det_arr.shape[1] >= 6:
                                for i in range(det_arr.shape[0]):
                                    x1, y1, x2, y2, score, cid = det_arr[i, 0:6].tolist()
                                    item = {
                                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                        "label": "unknown",
                                        "score": float(score),
                                        "track_id": None,
                                        "class_id": int(cid),
                                        "format": "xyxy",
                                    }
                                    _maybe_set_src_wh(item, cam, None)  # ← 핵심
                                    items.append(item)
                        except Exception:
                            pass

            overlays[cam_name] = items

        return overlays

    
    
    
    
    # ──────────────────────────────────────────────────────────────────────
    # 플랜 좌표 수집(기 엔진 제공) 또는 폴백 투영 계산
    # ──────────────────────────────────────────────────────────────────────
    def _collect_or_project_plan_points(
        self,
        result: Dict[str, Any],
        overlays: Dict[str, List[Dict[str, Any]]],
    ) -> Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]:
        """
        1) 엔진이 제공하는 cam['plan_coords'] 가 있으면 그대로 사용
        2) 없거나 비어 있으면 overlays의 bbox들을 호모그래피로 투영해 계산
        """
        # 1) 엔진 제공값 우선
        cams = result.get("cameras", []) or []
        has_any = any((cam.get("plan_coords") for cam in cams))
        if has_any:
            fused: List[Tuple[float, float]] = []
            coords_per_cam: List[List[Tuple[float, float]]] = []
            for cam in cams:
                pts = cam.get("plan_coords", []) or []
                coords_per_cam.append([(float(x), float(y)) for x, y in pts])
                fused.extend([(float(x), float(y)) for x, y in pts])
            return fused, coords_per_cam

        # 2) 폴백: bbox → 플랜 투영
        # 스트림 순서/이름 확보
        try:
            stream_names = list(self.tracking_system.streams.keys()) if self.tracking_system and self.tracking_system.streams else []
        except Exception:
            stream_names = []

        coords_per_cam: List[List[Tuple[float, float]]] = []
        fused: List[Tuple[float, float]] = []

        for cam_idx, cam_name in enumerate(stream_names):
            items = overlays.get(cam_name, []) or []
            H = self._get_homography_for_stream(cam_name)
            cam_pts: List[Tuple[float, float]] = []

            if H is None:
                # 호모그래피가 없으면 이 카메라는 스킵
                coords_per_cam.append(cam_pts)
                continue

            for it in items:
                score = float(it.get("score", 1.0))
                if score < self.min_score_for_projection:
                    continue
                bbox = it.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox[:4]
                if self.project_footpoint == "center":
                    px = (float(x1) + float(x2)) * 0.5
                    py = (float(y1) + float(y2)) * 0.5
                else:  # bottom_center
                    px = (float(x1) + float(x2)) * 0.5
                    py = float(y2)

                uv = self._project_point(H, px, py)
                if uv is not None:
                    cam_pts.append(uv)
                    fused.append(uv)

            coords_per_cam.append(cam_pts)

        return fused, coords_per_cam

    def _get_homography_for_stream(self, cam_name: str):
        """
        스트림 객체에서 호모그래피(이미지→플랜) 행렬(3x3)을 찾아 반환.
        streamSCST의 self.H 필드를 우선적으로 찾음.
        """
        try:
            stream = None
            if self.tracking_system and getattr(self.tracking_system, "streams", None):
                stream = self.tracking_system.streams.get(cam_name)
            if stream is None:
                print(f"[DEBUG] stream not found for cam_name: {cam_name}")
                return None

            # streamSCST의 H 필드를 직접 확인
            H = getattr(stream, "H", None)
            if H is not None:
                import numpy as np
                H = np.asarray(H, dtype=float)
                if H.shape == (3, 3):
                    print(f"[DEBUG] found H matrix for {cam_name}: shape={H.shape}")
                    return H
                else:
                    print(f"[DEBUG] H matrix shape invalid for {cam_name}: {H.shape}")

            # 폴백: 다른 필드명들도 시도
            candidates = ("homography", "H_cam2plan", "H_img2plan", "H_persp")
            for name in candidates:
                H = getattr(stream, name, None)
                if H is not None:
                    import numpy as np
                    H = np.asarray(H, dtype=float)
                    if H.shape == (3, 3):
                        print(f"[DEBUG] found {name} matrix for {cam_name}: shape={H.shape}")
                        return H

            print(f"[DEBUG] no valid homography matrix found for {cam_name}")
            return None
        except Exception as e:
            print(f"[DEBUG] error getting homography for {cam_name}: {e}")
            return None

    def _project_point(self, H, x: float, y: float) -> Optional[Tuple[float, float]]:
        """호모그래피 H로 (x,y,1) → (u,v) 투영"""
        try:
            import numpy as np
            vec = np.array([x, y, 1.0], dtype=float)
            uvw = H @ vec
            if abs(uvw[2]) < 1e-6:
                return None
            u = float(uvw[0] / uvw[2])
            v = float(uvw[1] / uvw[2])
            return (u, v)
        except Exception:
            return None

    # ──────────────────────────────────────────────────────────────────────
    # 종료
    # ──────────────────────────────────────────────────────────────────────
    def cleanup(self):
        print("🧹 리소스 정리 중...")
        self.web_manager.stop_web_server()
        if self.tracking_system:
            try:
                self.tracking_system.cleanup()
            except Exception:
                pass
        print("✅ 정리 완료")
