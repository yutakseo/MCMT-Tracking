# /workspace/tools/mapping.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import math
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 옵션
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class MatchConfig:
    # intra-round(동일 라운드) 카메라 간 클러스터링 게이트 (px, plan 좌표 기준)
    pair_gate: float = 120.0
    # tracks(글로벌 트랙) ↔ 현재 라운드 클러스터 매칭 게이트
    track_gate: float = 150.0
    # 트랙이 사라졌다가 몇 라운드까지 버틸지
    max_age: int = 10
    # 새 트랙이 확정되기까지 최소 연속 히트 횟수
    min_hits: int = 1

# ──────────────────────────────────────────────────────────────────────────────
# 간단 헝가리안 (scipy 없으면 그리디 폴백)
# ──────────────────────────────────────────────────────────────────────────────
def hungarian(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return row_ind, col_ind. If scipy 없으면 간단 그리디."""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost)
    except Exception:
        # Greedy fallback (정확 최적해 보장은 없음)
        n_rows, n_cols = cost.shape
        rows = set(range(n_rows))
        cols = set(range(n_cols))
        pairs = []
        while rows and cols:
            # 현재 남은 쌍 중 최저 코스트 선택
            r, c = min(((r, c) for r in rows for c in cols), key=lambda rc: cost[rc])
            pairs.append((r, c))
            rows.remove(r); cols.remove(c)
        if not pairs:
            return np.array([], dtype=int), np.array([], dtype=int)
        r_idx, c_idx = zip(*pairs)
        return np.array(r_idx, dtype=int), np.array(c_idx, dtype=int)

# ──────────────────────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────────────────────
def dist(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def centroid(points: List[Tuple[float,float]]) -> Tuple[float,float]:
    if not points: return (float('nan'), float('nan'))
    xs, ys = zip(*points)
    return (sum(xs)/len(xs), sum(ys)/len(ys))

# ──────────────────────────────────────────────────────────────────────────────
# 글로벌 트랙
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Track:
    track_id: int
    last_ts: float
    age: int = 0
    hits: int = 0
    missed: int = 0
    # 최근 클러스터 중심 (plan 좌표)
    center: Tuple[float,float] = (float('nan'), float('nan'))
    # 최근 속도(단순): center 변화량 / dt
    velocity: Tuple[float,float] = (0.0, 0.0)
    # 최근 멤버: { cam_idx: (x,y) }
    members: Dict[int, Tuple[float,float]] = field(default_factory=dict)

    def predict(self, ts: float) -> Tuple[float,float]:
        dt = max(ts - self.last_ts, 0.0)
        return (self.center[0] + self.velocity[0]*dt, self.center[1] + self.velocity[1]*dt)

    def update(self, ts: float, members: Dict[int, Tuple[float,float]], new_center: Tuple[float,float]):
        dt = max(ts - self.last_ts, 1e-3)
        vx = (new_center[0] - self.center[0]) / dt if math.isfinite(self.center[0]) else 0.0
        vy = (new_center[1] - self.center[1]) / dt if math.isfinite(self.center[1]) else 0.0
        self.velocity = (vx, vy)
        self.center = new_center
        self.members = members
        self.last_ts = ts
        self.hits += 1
        self.missed = 0
        self.age += 1

    def mark_missed(self):
        self.missed += 1
        self.age += 1

# ──────────────────────────────────────────────────────────────────────────────
# 온라인 멀티-카메라 매칭기
# ──────────────────────────────────────────────────────────────────────────────
class OnlineMultiCamMatcher:
    def __init__(self, cfg: Optional[MatchConfig] = None):
        self.cfg = cfg or MatchConfig()
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}  # active tracks

    # Step A) 동일 라운드에서 카메라 n대의 점들을 재귀적으로 매칭해 "클러스터" 구성
    def _cluster_cameras(self, coords_per_cam: List[List[Tuple[float,float]]]) -> List[Dict[str,Any]]:
        """
        coords_per_cam: 길이 = 카메라 수. 각 원소는 [(x,y), ...].
        return: clusters = [ {"members": {cam_idx: (x,y)}, "center": (cx,cy)} , ... ]
        """
        if not coords_per_cam:
            return []

        # 시작은 첫 번째 카메라 관측치 각각을 독립 클러스터로 둔다.
        clusters: List[Dict[str,Any]] = []
        for p in coords_per_cam[0]:
            clusters.append({"members": {0: p}, "center": p})

        # 2번째 카메라부터 순차 결합
        for cam_idx in range(1, len(coords_per_cam)):
            obs = coords_per_cam[cam_idx]
            if not obs:
                continue

            # 클러스터 중심 vs 관측점 거리 행렬
            centers = [c["center"] for c in clusters]
            if centers and obs:
                C = np.full((len(centers), len(obs)), fill_value=1e9, dtype=float)
                for i, c in enumerate(centers):
                    for j, p in enumerate(obs):
                        d = dist(c, p)
                        if d <= self.cfg.pair_gate:
                            C[i, j] = d
                r_idx, c_idx = hungarian(C)

                # 매칭된 것은 병합
                matched_obs = set()
                for ri, cj in zip(r_idx, c_idx):
                    if C[ri, cj] >= 1e8:
                        continue  # 게이트 밖
                    p = obs[cj]
                    clusters[ri]["members"][cam_idx] = p
                    # 중심 재계산
                    pts = list(clusters[ri]["members"].values())
                    clusters[ri]["center"] = centroid(pts)
                    matched_obs.add(cj)

                # 매칭 안된 관측은 신규 클러스터 생성
                for j, p in enumerate(obs):
                    if j not in matched_obs:
                        clusters.append({"members": {cam_idx: p}, "center": p})

        return clusters

    # Step B) 현재 라운드의 클러스터들을 기존 글로벌 트랙과 매칭 & 갱신
    def update(self, round_id: int, ts: float, coords_per_cam: List[List[Tuple[float,float]]]) -> Dict[str,Any]:
        """
        returns summary:
        {
          "round": int,
          "clusters": [ {"center":(x,y), "members":{cam_idx:(x,y)}} , ... ],
          "tracks":   [ {"id":tid, "center":(x,y), "members":{...}, "age":age, "hits":hits, "missed":missed} , ... ],
          "unmatched_clusters": [indices...],
          "unmatched_tracks":   [track_ids...],
        }
        """
        clusters = self._cluster_cameras(coords_per_cam)

        # 준비: 트랙 예측 위치 & 코스트 행렬
        track_ids = list(self.tracks.keys())
        pred_centers = [self.tracks[tid].predict(ts) for tid in track_ids]
        cl_centers = [c["center"] for c in clusters]

        # 매칭 없으면 바로 처리
        if not track_ids or not clusters:
            # 기존 트랙 age/미스 증가
            for t in self.tracks.values():
                t.mark_missed()
            # 미스 과다 제거
            self._remove_dead()
            # 새 클러스터는 모두 신규 트랙 생성
            for c in clusters:
                self._spawn_track(ts, c)
            return self._summary(round_id, clusters)

        # 비용행렬 (게이트 밖은 큰 수)
        C = np.full((len(track_ids), len(cl_centers)), fill_value=1e6, dtype=float)
        for i, pc in enumerate(pred_centers):
            for j, cc in enumerate(cl_centers):
                d = dist(pc, cc)
                if d <= self.cfg.track_gate:
                    C[i, j] = d

        r_idx, c_idx = hungarian(C)

        matched_tr = set()
        matched_cl = set()

        # 매칭 갱신
        for ri, cj in zip(r_idx, c_idx):
            if C[ri, cj] >= 1e5:
                continue  # 게이트 밖
            tid = track_ids[ri]
            cl = clusters[cj]
            self.tracks[tid].update(ts, cl["members"], cl["center"])
            matched_tr.add(tid)
            matched_cl.add(cj)

        # 매칭 안된 트랙 → missed
        for tid in track_ids:
            if tid not in matched_tr:
                self.tracks[tid].mark_missed()

        # 매칭 안된 클러스터 → 신규 트랙 생성
        for j, cl in enumerate(clusters):
            if j not in matched_cl:
                self._spawn_track(ts, cl)

        # 죽은 트랙 제거
        self._remove_dead()

        return self._summary(round_id, clusters)

    # ── 내부: 트랙 관리 ──────────────────────────────────────────────────────
    def _spawn_track(self, ts: float, cluster: Dict[str,Any]):
        tid = self._next_id; self._next_id += 1
        t = Track(track_id=tid, last_ts=ts, center=cluster["center"], members=cluster["members"])
        t.hits = 1; t.age = 1; t.missed = 0
        self.tracks[tid] = t

    def _remove_dead(self):
        dead = [tid for tid, t in self.tracks.items() if t.missed > self.cfg.max_age]
        for tid in dead: del self.tracks[tid]

    # ── 요약 포맷 ────────────────────────────────────────────────────────────
    def _summary(self, round_id: int, clusters: List[Dict[str,Any]]) -> Dict[str,Any]:
        tracks_out = []
        for t in self.tracks.values():
            if t.hits >= self.cfg.min_hits:
                tracks_out.append({
                    "id": t.track_id,
                    "center": t.center,
                    "members": dict(t.members),
                    "age": t.age,
                    "hits": t.hits,
                    "missed": t.missed,
                })
        # unmatched 정보(디버그 용)
        cl_member_sets = [set(c["members"].keys()) for c in clusters]
        # 간단히 남김 (상세 필요하면 로직 확장 가능)
        return {
            "round": round_id,
            "clusters": clusters,
            "tracks": tracks_out,
        }
