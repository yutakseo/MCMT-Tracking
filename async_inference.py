# /workspace/async_inference.py
from __future__ import annotations
import asyncio, logging, time, functools, inspect
from typing import List, Tuple, Any, Optional, Callable, Awaitable, Union
import numpy as np
import cv2
from collections import deque

from tools.SCST import SCST

# ── 고정 상수(자주 안 바꿈) ──────────────────────────────────────────────
WARMUP_FRAMES              = 12
WARMUP_TIMEOUT             = 5.0
CAPTURE_TIMEOUT_S          = 2.5
MAX_CAPTURE_SKEW_MS: float = 150.0
INTERVAL_BETWEEN_ROUND_S   = 0.0

# 반환 패킷 포맷:
# {"round": int, "coords": List[List[(x,y)]], "timestamps": List[float]}
RoundPacket = dict


class AsyncEngine:
    """
    사용 예:
        engine = AsyncEngine(cams, history_len=5)

        # 1) 무한 루프 모드 (Ctrl+C 로 종료):
        history = engine.run_sync()        # 종료 시 최신→오래된 순 리스트 반환
        print(engine.result)               # {round: [cam1_pts, cam2_pts, cam3_pts]} (최신 1건)
        print(history[:1])                 # 가장 최신 라운드 맵핑 프리뷰

        # 2) 배치 모드 (정확히 N 라운드만):
        pkts = engine.run_sync(max_rounds=20)   # RoundPacket 리스트(오래된→최신)
        print(engine.result_history[0])         # 최신 라운드 맵핑
    """

    def __init__(
        self,
        cams: List[SCST],
        on_round: Optional[Callable[[RoundPacket], Awaitable[None] | None]] = None,
        history_len: int = 10,  # 저장할 히스토리 개수(최신→오래된), 최소 1
    ):
        self.cams: List[SCST] = cams
        self._on_round = on_round
        self._stop_evt: Optional[asyncio.Event] = None

        # 최신 결과(항상 1건) — {round: [cam1_pts, cam2_pts, cam3_pts]}
        self.result: Optional[dict[int, List[List[Tuple[float, float]]]]] = None

        # 히스토리(최신→오래된). 요소는 self.result와 같은 "mapping" 형태
        if history_len <= 0:
            history_len = 1
        self._history_len = history_len
        self.result_history: deque[dict[int, List[List[Tuple[float, float]]]]] = deque(maxlen=self._history_len)

        # 마지막 라운드의 풀 패킷(RoundPacket: round/coords/timestamps)
        self.last_packet: Optional[RoundPacket] = None

    # ── Public API ────────────────────────────────────────────────────────────
    def set_on_round(self, cb: Callable[[RoundPacket], Awaitable[None] | None]) -> None:
        self._on_round = cb

    async def run(self, *, max_rounds: Optional[int] = None) -> Union[RoundPacket, List[RoundPacket], List[dict]]:
        # 실행 전에 히스토리만 초기화 (크기 maxlen 유지)
        self.result_history.clear()

        await self._warmup_all()
        self._stop_evt = asyncio.Event()

        collected: List[RoundPacket] = []  # 배치 모드 수집(오래된→최신)
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

                # 추론 (순차)
                coords_per_cam = await self._inference_round(frames)

                # 로그(프리뷰)
                for idx, pts in enumerate(coords_per_cam, 1):
                    preview = [(round(x, 1), round(y, 1)) for x, y in pts[:10]]
                    logging.info(f"[round {r}] cam{idx} plan coords: {preview if pts else []}")
                if r % 5 == 0:
                    logging.info(f"[round {r}] inference done for cam1, cam2, cam3")

                # 패킷 & 필드 갱신
                pkt: RoundPacket = {"round": r, "coords": coords_per_cam, "timestamps": ts}
                self.last_packet = pkt

                mapping = {r: coords_per_cam}  # 요구 포맷
                self.result = mapping                          # 최신 1건 덮어쓰기
                self.result_history.appendleft(mapping)        # 최신→오래된, 길이 history_len 유지

                # 콜백
                if self._on_round is not None:
                    maybe_coro = self._on_round(pkt)
                    if asyncio.iscoroutine(maybe_coro):
                        await maybe_coro

                # 배치 모드: N 라운드 수집되면 반환
                if max_rounds is not None:
                    collected.append(pkt)
                    if len(collected) >= max_rounds:
                        return collected  # 오래된→최신

                if INTERVAL_BETWEEN_ROUND_S > 0:
                    await asyncio.sleep(INTERVAL_BETWEEN_ROUND_S)

                r += 1
        finally:
            self._stop_evt = None

        # 무한루프 모드 종료 시: 최신→오래된 순의 히스토리(list) 반환
        return list(self.result_history)

    def run_sync(self, *, max_rounds: Optional[int] = None) -> Union[RoundPacket, List[RoundPacket], List[dict]]:
        try:
            return asyncio.run(self.run(max_rounds=max_rounds))
        except KeyboardInterrupt:
            # 무한루프/배치 모두에서 뭔가 반환
            return list(self.result_history) if self.result_history else (
                self.last_packet or {"round": -1, "coords": [[], [], []], "timestamps": []}
            )

    def stop(self) -> None:
        if self._stop_evt is not None:
            self._stop_evt.set()

    # ── Internals ────────────────────────────────────────────────────────────
    @staticmethod
    async def _run_blocking(fn, *a, **kw):
        try:
            to_thread = asyncio.to_thread  # py>=3.9
        except AttributeError:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, functools.partial(fn, *a, **kw))
        else:
            return await to_thread(fn, *a, **kw)

    @staticmethod
    def _is_valid_frame(x) -> bool:
        return isinstance(x, np.ndarray) and x.ndim in (2, 3) and x.size > 0

    @staticmethod
    def _extract_frame_from(cam: SCST, ret) -> Optional[np.ndarray]:
        if isinstance(ret, np.ndarray):
            return ret
        if isinstance(ret, (tuple, list)):
            for item in ret:
                if isinstance(item, np.ndarray):
                    return item
        for name in ("frame", "_frame", "latest_frame", "_latest_frame"):
            f = getattr(cam, name, None)
            if isinstance(f, np.ndarray):
                return f
        return None

    @staticmethod
    def _inject_frame(cam: SCST, frame: np.ndarray) -> None:
        for name in ("frame", "_frame", "latest_frame", "_latest_frame"):
            setattr(cam, name, frame)
        trk = getattr(cam, "tracker", None)
        if trk is not None:
            for name in ("frame", "_frame", "latest_frame", "_latest_frame"):
                try:
                    setattr(trk, name, frame)
                except Exception:
                    pass

    @staticmethod
    def _get_plan_wh(cam: SCST) -> Optional[Tuple[int, int]]:
        proj = getattr(cam, "projector", None)
        if proj is None:
            return None
        for attr in ("plan_img", "_plan_img", "img", "plan"):
            pl = getattr(proj, attr, None)
            if isinstance(pl, np.ndarray) and pl.ndim >= 2:
                return int(pl.shape[1]), int(pl.shape[0])
        # width/height 속성 보유 케이스
        w = None
        for k in ("width", "plan_w", "W"):
            val = getattr(proj, k, None)
            if val is not None:
                w = val
                break
        h = None
        for k in ("height", "plan_h", "H"):
            val = getattr(proj, k, None)
            if val is not None:
                h = val
                break
        try:
            return (int(w), int(h)) if (w is not None and h is not None) else None
        except Exception:
            return None

    @staticmethod
    def _in_bounds(pts: List[Tuple[float, float]], plan_wh: Optional[Tuple[int, int]]) -> List[Tuple[float, float]]:
        if not pts:
            return []
        if plan_wh is None:
            return [(x, y) for x, y in pts if np.isfinite(x) and np.isfinite(y)]
        W, H = plan_wh
        return [(x, y) for x, y in pts if np.isfinite(x) and np.isfinite(y) and 0 <= x <= W and 0 <= y <= H]

    @staticmethod
    def _parse_plan_points(results: Any) -> List[Tuple[float, float]]:
        if results is None:
            return []
        if isinstance(results, np.ndarray) and results.ndim == 2 and results.shape[1] >= 2:
            return [(float(results[i, 0]), float(results[i, 1])) for i in range(results.shape[0])]
        pts: List[Tuple[float, float]] = []
        if isinstance(results, (list, tuple)):
            for it in results:
                if isinstance(it, dict):
                    if "plan" in it and isinstance(it["plan"], (list, tuple, np.ndarray)) and len(it["plan"]) >= 2:
                        pts.append((float(it["plan"][0]), float(it["plan"][1])))
                        continue
                    for k in ("pt", "xy", "coord"):
                        if k in it and isinstance(it[k], (list, tuple, np.ndarray)) and len(it[k]) >= 2:
                            pts.append((float(it[k][0]), float(it[k][1])))
                            break
                    else:
                        if "x" in it and "y" in it:
                            pts.append((float(it["x"]), float(it["y"])))
                elif isinstance(it, (list, tuple, np.ndarray)) and len(it) >= 2:
                    pts.append((float(it[0]), float(it[1])))
        return pts

    @staticmethod
    def _parse_img_points_from_tracklets(tracklets: Any) -> List[Tuple[float, float]]:
        if tracklets is None:
            return []
        if isinstance(tracklets, np.ndarray):
            arr = tracklets
            if arr.ndim == 2 and arr.shape[1] >= 4:
                x1 = arr[:, 0].astype(float)
                y1 = arr[:, 1].astype(float)
                x2 = arr[:, 2].astype(float)
                y2 = arr[:, 3].astype(float)
                return list(zip(((x1 + x2) * 0.5).tolist(), y2.tolist()))
            if arr.ndim == 2 and arr.shape[1] == 2:
                return [(float(a), float(b)) for a, b in arr]
            return []
        pts: List[Tuple[float, float]] = []
        if isinstance(tracklets, (list, tuple)):
            for t in tracklets:
                if isinstance(t, dict):
                    for k in ("bbox", "box", "tlbr", "xyxy"):
                        if k in t and isinstance(t[k], (list, tuple, np.ndarray)) and len(t[k]) >= 4:
                            x1, y1, x2, y2 = map(float, t[k][:4])
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

    @staticmethod
    def _warp_with_H(H: Optional[np.ndarray], img_pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if H is None or not isinstance(H, np.ndarray) or H.shape != (3, 3) or not img_pts:
            return []
        pts = np.array(img_pts, dtype=np.float32).reshape(-1, 1, 2)
        try:
            warped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        except Exception:
            return []
        return [(float(x), float(y)) for x, y in warped]

    # ── 단계별 작업 ───────────────────────────────────────────────────────────
    async def _warmup_camera(self, cam: SCST, frames: int = WARMUP_FRAMES, timeout: float = WARMUP_TIMEOUT) -> bool:
        start, got = time.time(), 0
        while got < frames and (time.time() - start) < timeout:
            try:
                ret = await self._run_blocking(cam._videoCapture)
                if self._is_valid_frame(self._extract_frame_from(cam, ret)):
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
        tasks = [self._run_blocking(cam._videoCapture) for cam in self.cams]
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
            f = self._extract_frame_from(cam, r)
            if not self._is_valid_frame(f):
                logging.error(f"capture_round: cam{i} invalid frame")
                ok = False
                frames.append(None)
            else:
                frames.append(f)
        ts = [time.time()] * len(self.cams)
        return ok, frames, ts

    async def _try_inference(self, cam: SCST, frame: np.ndarray) -> Optional[Tuple[Any, Any]]:
        try:
            sig = inspect.signature(cam.inference)
            if "frame" in sig.parameters:
                out = await self._run_blocking(cam.inference, frame)
                if isinstance(out, tuple) and len(out) == 2:
                    return out
        except Exception:
            pass
        if hasattr(cam, "inference_once"):
            try:
                out = await self._run_blocking(cam.inference_once, frame)
                if isinstance(out, tuple) and len(out) == 2:
                    return out
            except Exception:
                pass
        try:
            self._inject_frame(cam, frame)
            out = await self._run_blocking(cam.inference)
            if isinstance(out, tuple) and len(out) == 2:
                return out
        except Exception:
            pass
        return None

    async def _safe_infer(self, cam: SCST, frame: Optional[np.ndarray]) -> List[Tuple[float, float]]:
        if not self._is_valid_frame(frame):
            raise TypeError("safe_infer(): invalid frame")

        plan_wh = self._get_plan_wh(cam)

        out = await self._try_inference(cam, frame)
        if out is not None:
            _proj_img, results = out
            pts = self._in_bounds(self._parse_plan_points(results), plan_wh)
            if pts:
                return pts
            logging.warning("safe_infer: projector out-of-bounds/empty -> []")
            return []

        tracklets = getattr(cam, "tracklets", None)
        if tracklets is None:
            trk = getattr(cam, "tracker", None)
            if trk is not None and hasattr(trk, "track_image"):
                try:
                    tracklets = await self._run_blocking(trk.track_image, frame=frame, visualize=False)
                except Exception as e:
                    logging.warning(f"safe_infer: track_image failed: {e}")
        img_pts = self._parse_img_points_from_tracklets(tracklets)
        if img_pts:
            warped = self._in_bounds(self._warp_with_H(getattr(cam, "H", None), img_pts), plan_wh)
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
