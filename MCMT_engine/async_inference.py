# /workspace/async_inference.py
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import time
from collections import deque
from typing import Any, Awaitable, Callable, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from MCMT_engine.SCST import SCST

# 매칭기 (사용자 모듈)
from MCMT_engine.mapping import OnlineMultiCamMatcher, MatchConfig

# ── 고정 상수 ─────────────────────────────────────────────────────────────
WARMUP_FRAMES = 12
WARMUP_TIMEOUT = 5.0
CAPTURE_TIMEOUT_S = 2.5
MAX_CAPTURE_SKEW_MS: float = 150.0
INTERVAL_BETWEEN_ROUND_S = 0.0  # 라운드 사이 sleep

# RoundPacket: 1 라운드의 결과 패킷
# {"round": int, "coords": List[List[(x,y)]], "timestamps": List[float], "fused": List[(x,y)]}
RoundPacket = Dict[str, Any]

# ── 유틸 ────────────────────────────────────────────────────────────────────
async def run_blocking(fn: Callable, *a, **kw):
    """동기 함수를 스레드로 실행 (py3.8 호환)."""
    try:
        to_thread = asyncio.to_thread  # py>=3.9
    except AttributeError:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *a, **kw))
    else:
        return await to_thread(fn, *a, **kw)

def is_valid_frame(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.ndim in (2, 3) and x.size > 0

def extract_frame(cam: SCST, ret: Any) -> Optional[np.ndarray]:
    """SCST._videoCapture 반환/속성에서 프레임을 최대한 뽑아낸다."""
    if isinstance(ret, np.ndarray):
        return ret
    if isinstance(ret, (tuple, list)):
        for it in ret:
            if isinstance(it, np.ndarray):
                return it
    for name in ("frame", "_frame", "latest_frame", "_latest_frame"):
        f = getattr(cam, name, None)
        if isinstance(f, np.ndarray):
            return f
    return None

def inject_frame(cam: SCST, frame: np.ndarray) -> None:
    """SCST/트래커가 어떤 속성을 참조하든 프레임을 찾도록 주입."""
    for name in ("frame", "_frame", "latest_frame", "_latest_frame"):
        setattr(cam, name, frame)
    trk = getattr(cam, "tracker", None)
    if trk is not None:
        for name in ("frame", "_frame", "latest_frame", "_latest_frame"):
            try:
                setattr(trk, name, frame)
            except Exception:
                pass

def get_plan_wh(cam: SCST) -> Optional[Tuple[int, int]]:
    """Plan 이미지의 (W,H) 추정."""
    proj = getattr(cam, "projector", None)
    if proj is None:
        return None
    for attr in ("plan_img", "_plan_img", "img", "plan"):
        pl = getattr(proj, attr, None)
        if isinstance(pl, np.ndarray) and pl.ndim >= 2:
            return int(pl.shape[1]), int(pl.shape[0])
    w = next((getattr(proj, k, None) for k in ("width", "plan_w", "W") if getattr(proj, k, None) is not None), None)
    h = next((getattr(proj, k, None) for k in ("height", "plan_h", "H") if getattr(proj, k, None) is not None), None)
    try:
        return (int(w), int(h)) if (w is not None and h is not None) else None
    except Exception:
        return None

def in_bounds(pts: List[Tuple[float, float]], plan_wh: Optional[Tuple[int, int]]) -> List[Tuple[float, float]]:
    """플랜 경계 내 점만 유지. plan_wh 없으면 NaN/Inf만 필터."""
    if not pts:
        return []
    if plan_wh is None:
        return [(x, y) for (x, y) in pts if np.isfinite(x) and np.isfinite(y)]
    W, H = plan_wh
    return [(x, y) for (x, y) in pts if np.isfinite(x) and np.isfinite(y) and 0 <= x <= W and 0 <= y <= H]

def parse_plan_points(results: Any) -> List[Tuple[float, float]]:
    """projector.projection(..)의 results에서 (x,y) 리스트 파싱."""
    if results is None:
        return []
    if isinstance(results, np.ndarray) and results.ndim == 2 and results.shape[1] >= 2:
        return [(float(results[i, 0]), float(results[i, 1])) for i in range(results.shape[0])]
    out: List[Tuple[float, float]] = []
    if isinstance(results, (list, tuple)):
        for it in results:
            if isinstance(it, dict):
                if "plan" in it and isinstance(it["plan"], (list, tuple, np.ndarray)) and len(it["plan"]) >= 2:
                    out.append((float(it["plan"][0]), float(it["plan"][1])))
                    continue
                for k in ("pt", "xy", "coord"):
                    v = it.get(k)
                    if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
                        out.append((float(v[0]), float(v[1])))
                        break
                else:
                    if "x" in it and "y" in it:
                        out.append((float(it["x"]), float(it["y"])))
            elif isinstance(it, (list, tuple, np.ndarray)) and len(it) >= 2:
                out.append((float(it[0]), float(it[1])))
    return out

def parse_img_points_from_tracklets(tracklets: Any) -> List[Tuple[float, float]]:
    """트랙렛에서 바텀센터 등의 이미지 좌표 파싱."""
    if tracklets is None:
        return []
    # ndarray
    if isinstance(tracklets, np.ndarray):
        arr = tracklets
        if arr.ndim == 2 and arr.shape[1] >= 4:
            x1, y1, x2, y2 = arr[:, 0].astype(float), arr[:, 1].astype(float), arr[:, 2].astype(float), arr[:, 3].astype(float)
            return list(zip(((x1 + x2) * 0.5).tolist(), y2.tolist()))
        if arr.ndim == 2 and arr.shape[1] == 2:
            return [(float(a), float(b)) for a, b in arr]
        return []
    # list-like
    pts: List[Tuple[float, float]] = []
    if isinstance(tracklets, (list, tuple)):
        for t in tracklets:
            if isinstance(t, dict):
                for k in ("bbox", "box", "tlbr", "xyxy"):
                    v = t.get(k)
                    if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 4:
                        x1, y1, x2, y2 = map(float, v[:4])
                        pts.append(((x1 + x2) * 0.5, y2))
                        break
                else:
                    if "x" in t and "y" in t:
                        pts.append((float(t["x"]), float(t["y"])))
            elif isinstance(t, (list, tuple, np.ndarray)):
                if len(t) >= 4:
                    x1, y1, x2, y2 = map(float, t[:4])
                    pts.append(((x1 + x2) * 0.5, y2))
                elif len(t) >= 2:
                    pts.append((float(t[0]), float(t[1])))
    return pts

def warp_with_H(H: Optional[np.ndarray], img_pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """호모그래피로 이미지 좌표를 플랜 좌표로 워핑."""
    if H is None or not isinstance(H, np.ndarray) or H.shape != (3, 3) or not img_pts:
        return []
    pts = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
    try:
        warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    except Exception:
        return []
    return [(float(x), float(y)) for x, y in warped]

# ── AsyncEngine ─────────────────────────────────────────────────────────────
class AsyncEngine:
    """
    • async 스트리밍(권장):
        async for pkt in engine.stream():
            print(pkt["fused"])  # 매 라운드 정합 좌표

    • 배치/테스트:
        pkts = engine.run_sync(max_rounds=20)  # 20라운드 수집 후 반환

    상태 확인:
        engine.result         # {round: [cam1_pts, cam2_pts, ...]} (최신 1건)
        engine.result_history # deque 최신→오래된, 길이 history_len 유지
        engine.last_packet    # 마지막 RoundPacket (round/coords/timestamps/fused)
        engine.fused_latest   # 최신 라운드의 정합 좌표 리스트
    """
    def __init__(
        self,
        cams: List[SCST],
        on_round: Optional[Callable[[RoundPacket], Awaitable[None] | None]] = None,
        history_len: int = 10,
    ):
        self.cams: List[SCST] = cams
        self._on_round = on_round
        self._stop_evt: Optional[asyncio.Event] = None

        # 최신 1건: {round: [cam1_pts, cam2_pts, ...]}
        self.result: Optional[Dict[int, List[List[Tuple[float, float]]]]] = None
        # 최신→오래된: 길이 history_len 유지
        self.result_history: Deque[Dict[int, List[List[Tuple[float, float]]]]] = deque(maxlen=max(1, history_len))
        # 마지막 패킷
        self.last_packet: Optional[RoundPacket] = None

        # 온라인 매칭기
        self.matcher = OnlineMultiCamMatcher(MatchConfig(
            pair_gate=120.0,
            track_gate=150.0,
            max_age=10,
            min_hits=1,
        ))
        self.tracks_snapshot: Optional[Dict[str, Any]] = None
        self.tracks_history: Deque[Dict[str, Any]] = deque(maxlen=max(1, history_len))

        # 최신 fused 좌표
        self.fused_latest: Optional[List[Tuple[float, float]]] = None

    # ── Public API ──────────────────────────────────────────────────────────
    def set_on_round(self, cb: Callable[[RoundPacket], Awaitable[None] | None]) -> None:
        self._on_round = cb

    async def stream(self, *, max_rounds: Optional[int] = None):
        """
        라운드마다 RoundPacket을 yield 하는 async generator.
        무한 스트림 모드(기본) 또는 max_rounds 만큼만 배출.
        """
        # 실행 전 히스토리만 비움
        self.result_history.clear()

        await self._warmup_all()
        self._stop_evt = asyncio.Event()

        r = 0
        try:
            while not self._stop_evt.is_set():
                ok, frames, ts = await self._capture_round()
                if not ok:
                    logging.warning(f"[round {r}] capture failed -> skip")
                    await asyncio.sleep(0.02)
                    r += 1
                    continue

                if MAX_CAPTURE_SKEW_MS is not None:
                    skew_ms = (max(ts) - min(ts)) * 1000.0
                    if skew_ms > MAX_CAPTURE_SKEW_MS:
                        logging.warning(f"[round {r}] capture skew {skew_ms:.1f} ms")

                coords_per_cam = await self._inference_round(frames)

                # ★ 수정: 라운드/타임스탬프/좌표 모두 전달
                fused = self._update_matching(r, ts, coords_per_cam)

                # 프리뷰 로그
                for idx, pts in enumerate(coords_per_cam, 1):
                    preview = [(round(x, 1), round(y, 1)) for x, y in pts[:5]]
                    logging.info(f"[round {r}] cam{idx} plan coords: {preview if pts else []}")
                if fused:
                    fprev = [(round(x, 1), round(y, 1)) for x, y in fused[:10]]
                    logging.info(f"[round {r}] FUSED coords: {fprev}")
                if r % 5 == 0:
                    logging.info(f"[round {r}] inference done for {len(coords_per_cam)} cams")

                # 패킷 & 필드 갱신
                pkt: RoundPacket = {"round": r, "coords": coords_per_cam, "timestamps": ts, "fused": fused}
                self.last_packet = pkt
                self.result = {r: coords_per_cam}
                self.result_history.appendleft(self.result)
                self.fused_latest = fused

                # 콜백
                if self._on_round is not None:
                    maybe_coro = self._on_round(pkt)
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro

                # 외부로 전달
                yield pkt

                # 배치 종료 조건
                if max_rounds is not None and r + 1 >= max_rounds:
                    break

                if INTERVAL_BETWEEN_ROUND_S > 0:
                    await asyncio.sleep(INTERVAL_BETWEEN_ROUND_S)

                r += 1
        finally:
            self._stop_evt = None

    async def run(self, *, max_rounds: Optional[int] = None) -> Union[List[RoundPacket], List[dict]]:
        """
        배치 실행: max_rounds 지정 시 해당 라운드 수집 후 리스트 반환.
        지정 안 하면(무한) Ctrl+C로 중단하면 히스토리(최신→오래된) 리스트 반환.
        """
        if max_rounds is not None:
            out: List[RoundPacket] = []
            async for pkt in self.stream(max_rounds=max_rounds):
                out.append(pkt)
            return out
        # 무한 스트림: 사용자가 KeyboardInterrupt 시점에 히스토리 반환
        try:
            async for _ in self.stream():
                pass
        except KeyboardInterrupt:
            pass
        return list(self.result_history)

    def run_sync(self, *, max_rounds: Optional[int] = None) -> Union[List[RoundPacket], List[dict]]:
        return asyncio.run(self.run(max_rounds=max_rounds))

    def stop(self) -> None:
        if self._stop_evt is not None:
            self._stop_evt.set()

    # ── 매칭 헬퍼 ───────────────────────────────────────────────────────────
    @staticmethod
    def _extract_fused_positions(snapshot: Dict[str, Any]) -> List[Tuple[float, float]]:
        """
        매칭 스냅샷에서 대표 플랜좌표를 추출.
        tracks[*].xy → tracks[*].state[:2] → clusters[*].centroid → clusters[*].members 평균
        """
        if not snapshot:
            return []
        fused: List[Tuple[float, float]] = []

        tracks = snapshot.get("tracks")
        if isinstance(tracks, list) and tracks:
            for t in tracks:
                if isinstance(t, dict):
                    if "xy" in t and isinstance(t["xy"], (list, tuple)) and len(t["xy"]) >= 2:
                        x, y = float(t["xy"][0]), float(t["xy"][1]); fused.append((x, y))
                    elif "state" in t and isinstance(t["state"], (list, tuple, np.ndarray)) and len(t["state"]) >= 2:
                        x, y = float(t["state"][0]), float(t["state"][1]); fused.append((x, y))
            if fused:
                return fused

        clusters = snapshot.get("clusters")
        if isinstance(clusters, list) and clusters:
            # 1) centroid 우선
            for c in clusters:
                if isinstance(c, dict) and "centroid" in c:
                    cen = c["centroid"]
                    if isinstance(cen, (list, tuple, np.ndarray)) and len(cen) >= 2:
                        fused.append((float(cen[0]), float(cen[1])))
            if fused:
                return fused
            # 2) 멤버 평균
            for c in clusters:
                if not isinstance(c, dict):
                    continue
                mems = c.get("members")
                if not isinstance(mems, list) or not mems:
                    continue
                xs = ys = 0.0; n = 0
                for m in mems:
                    if isinstance(m, (list, tuple)) and len(m) >= 4:
                        xs += float(m[2]); ys += float(m[3]); n += 1
                    elif isinstance(m, dict):
                        x, y = m.get("x"), m.get("y")
                        if x is not None and y is not None:
                            xs += float(x); ys += float(y); n += 1
                if n > 0:
                    fused.append((xs / n, ys / n))

        return fused

    def _update_matching(
        self,
        round_idx: int,
        ts: List[float],
        coords_per_cam: List[List[Tuple[float, float]]],
    ) -> List[Tuple[float, float]]:
        """
        매칭기의 update/step 시그니처를 동적으로 감지해서 안전하게 호출.
        지원 형태 예시:
        - update(coords_per_cam)
        - update(ts, coords_per_cam)
        - update(round, ts, coords_per_cam)
        - update(ts=..., coords_per_cam=...)
        - update(round_idx=..., timestamps=..., coords=...)
        실패 시 빈 스냅샷 처리.
        """
        snapshot: Dict[str, Any] = {}

        def _call_various(meth):
            # 1) 시그니처 기반 kwargs 구성
            try:
                sig = inspect.signature(meth)
                kwargs = {}
                for name in sig.parameters.keys():
                    if name in ("round", "round_idx", "r"):
                        kwargs[name] = round_idx
                    elif name in ("ts", "timestamps", "times"):
                        kwargs[name] = ts
                    elif name in ("coords_per_cam", "coords", "points", "detections"):
                        kwargs[name] = coords_per_cam
                if kwargs:
                    return meth(**kwargs)
            except Exception:
                pass
            # 2) 포지셔널 다양한 조합 시도 (가장 풍부한 → 간단한 순)
            for args in (
                (round_idx, ts, coords_per_cam),
                (ts, coords_per_cam),
                (coords_per_cam,),
            ):
                try:
                    return meth(*args)
                except Exception:
                    continue
            raise TypeError("no compatible signature for matcher method")

        try:
            m = self.matcher
            if hasattr(m, "update"):
                snapshot = _call_various(m.update)
            elif hasattr(m, "step"):
                snapshot = _call_various(m.step)
        except Exception as e:
            logging.error(f"matcher update failed: {e}")
            snapshot = {}

        # 스냅샷/히스토리 저장
        if not isinstance(snapshot, dict):
            snapshot = {}
        self.tracks_snapshot = snapshot
        self.tracks_history.appendleft(snapshot)

        # 대표 fused 좌표 추출 (tracks/centroid/멤버평균 등 유연 파서)
        fused = self._extract_fused_positions(snapshot)
        self.fused_latest = fused
        return fused

    # ── 단계별(웜업/캡처/추론) ───────────────────────────────────────────────
    async def _warmup_camera(self, cam: SCST, frames: int = WARMUP_FRAMES, timeout: float = WARMUP_TIMEOUT) -> bool:
        start, got = time.time(), 0
        while got < frames and (time.time() - start) < timeout:
            try:
                ret = await run_blocking(cam._videoCapture)
                if is_valid_frame(extract_frame(cam, ret)):
                    got += 1
            except Exception:
                pass
        (logging.info if got else logging.warning)(f"warmup_camera: got {got}/{frames} frames")
        return got > 0

    async def _warmup_all(self) -> None:
        res = await asyncio.gather(*[self._warmup_camera(c) for c in self.cams], return_exceptions=True)
        for i, r in enumerate(res, 1):
            if r is not True:
                logging.warning(f"warmup_all: cam{i} not ready (result={r})")

    async def _capture_round(self) -> Tuple[bool, List[Optional[np.ndarray]], List[float]]:
        t0 = time.time()
        tasks = [run_blocking(cam._videoCapture) for cam in self.cams]
        try:
            rets = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=CAPTURE_TIMEOUT_S)
        except asyncio.TimeoutError:
            logging.warning("capture_round: timeout")
            return False, [None] * len(self.cams), [t0] * len(self.cams)

        ok, frames = True, []
        for i, (cam, r) in enumerate(zip(self.cams, rets), 1):
            if isinstance(r, Exception):
                logging.error(f"capture_round: cam{i} error: {r}")
                ok = False
                frames.append(None)
                continue
            f = extract_frame(cam, r)
            if not is_valid_frame(f):
                logging.error(f"capture_round: cam{i} invalid frame")
                ok = False
                frames.append(None)
            else:
                frames.append(f)
        ts = [time.time()] * len(self.cams)
        return ok, frames, ts

    async def _try_inference(self, cam: SCST, frame: np.ndarray) -> Optional[Tuple[Any, Any]]:
        # 1) inference(frame=...)
        try:
            sig = inspect.signature(cam.inference)
            if "frame" in sig.parameters:
                out = await run_blocking(cam.inference, frame)
                if isinstance(out, tuple) and len(out) == 2:
                    return out
        except Exception:
            pass
        # 2) inference_once(frame)
        if hasattr(cam, "inference_once"):
            try:
                out = await run_blocking(cam.inference_once, frame)
                if isinstance(out, tuple) and len(out) == 2:
                    return out
            except Exception:
                pass
        # 3) 프레임 주입 후 inference()
        try:
            inject_frame(cam, frame)
            out = await run_blocking(cam.inference)
            if isinstance(out, tuple) and len(out) == 2:
                return out
        except Exception:
            pass
        return None

    async def _safe_infer(self, cam: SCST, frame: Optional[np.ndarray]) -> List[Tuple[float, float]]:
        if not is_valid_frame(frame):
            raise TypeError("safe_infer(): invalid frame")

        plan_wh = get_plan_wh(cam)

        # A) 정상 경로: projector 결과
        out = await self._try_inference(cam, frame)
        if out is not None:
            _proj_img, results = out
            pts = in_bounds(parse_plan_points(results), plan_wh)
            if pts:
                return pts
            logging.warning("safe_infer: projector out-of-bounds/empty -> []")
            return []

        # B) Fallback: 트랙렛 바텀센터 → H 워핑
        tracklets = getattr(cam, "tracklets", None)
        if tracklets is None:
            trk = getattr(cam, "tracker", None)
            if trk is not None and hasattr(trk, "track_image"):
                try:
                    tracklets = await run_blocking(trk.track_image, frame=frame, visualize=False)
                except Exception as e:
                    logging.warning(f"safe_infer: track_image failed: {e}")

        img_pts = parse_img_points_from_tracklets(tracklets)
        if img_pts:
            warped = in_bounds(warp_with_H(getattr(cam, "H", None), img_pts), plan_wh)
            if warped:
                logging.warning("safe_infer: using manual H warp")
                return warped
            logging.warning("safe_infer: manual H warp out-of-bounds/empty -> []")
            return []

        logging.warning("safe_infer: no tracklets -> []")
        return []

    async def _inference_round(self, frames: List[Optional[np.ndarray]]) -> List[List[Tuple[float, float]]]:
        out: List[List[Tuple[float, float]]] = []
        for i, (cam, f) in enumerate(zip(self.cams, frames), 1):
            try:
                pts = await self._safe_infer(cam, f)
            except Exception as e:
                logging.error(f"inference_round: cam{i} failed: {e}")
                pts = []
            out.append(pts)
        return out
