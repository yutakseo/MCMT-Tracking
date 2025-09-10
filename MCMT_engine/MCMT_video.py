# /workspace/MCMT_engine/MCMT_video.py
# Multi-Camera Multi-Target Video Processing System (Seocho-ready, per-camera plan points)

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys
sys.path.append("/workspace")  # 프로젝트 루트 보장

from MCMT_engine.streaming.video_SCST import videoSCST, Args
from tools.homo_graphy import PlanProjector


# ───────────────────────────────────────────────────────────────────────────────
# 추적 파라미터
# ───────────────────────────────────────────────────────────────────────────────
class VideoTrackingArgs(Args):
    track_thresh = 0.3
    match_thresh = 0.9
    track_buffer = 180
    mot20 = False
    cpu_workers = 20
    chunk_sec = 10.0
    batch_size = 20


# ───────────────────────────────────────────────────────────────────────────────
# 멀티카메라 시스템
# ───────────────────────────────────────────────────────────────────────────────
class MCMTVideoSystem:
    """
    - 카메라별로 '플랜(도면) 기준점'이 서로 달라도 사용 가능
    - 각 카메라별 SCST 인스턴스를 만들고, 비디오를 병렬/순차 처리
    - 카메라 뷰/도면 투영 결과 비디오 저장 + 간단 리포트 저장
    """

    def __init__(
        self,
        args: VideoTrackingArgs,
        plan_path: str,
        plan_points_per_camera: List[List[Tuple[float, float]]],
        video_paths: List[str],
        camera_points: List[List[Tuple[float, float]]],
        output_dir: str = "/workspace/results",
        detector_models: Optional[List[str]] = None,
    ):
        # 입력 검증
        if len(video_paths) != len(camera_points) or len(video_paths) != len(plan_points_per_camera):
            raise ValueError(
                "카메라 수/비디오 수/플랜포인트 묶음 수가 불일치: "
                f"videos={len(video_paths)}, cams={len(camera_points)}, plan_sets={len(plan_points_per_camera)}"
            )
        # 카메라별로 CCTV 포인트 수 == 해당 카메라 플랜 포인트 수
        for i, (cctv, plan) in enumerate(zip(camera_points, plan_points_per_camera)):
            if len(cctv) != len(plan):
                raise ValueError(f"[Cam {i+1}] CCTV 포인트({len(cctv)}) != PLAN 포인트({len(plan)})")

        self.args = args
        self.plan_path: Path = Path(plan_path)
        self.plan_points_per_camera = plan_points_per_camera
        self.video_paths = video_paths
        self.camera_points = camera_points
        self.output_dir: Path = Path(output_dir)
        self.detector_models = detector_models

        # 출력 디렉토리 준비
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 컨테이너
        self.results: List[List[Dict[str, Any]]] = []  # 카메라별 프레임 결과 모음
        self.scst_instances: List[videoSCST] = []

        # 로깅
        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(asctime)s - %(message)s",
            datefmt="%H:%M:%S",
        )

        # 컴포넌트 초기화
        self._init_components()

    # ───────────────────────────────────────────────────────────────────
    # 초기화/생성
    # ───────────────────────────────────────────────────────────────────
    def _init_components(self) -> None:
        logging.info("컴포넌트 초기화 중...")

        # (1) 통합 플랜 비디오용 공유 Projector (플랜 이미지만 공유)
        self.projector = PlanProjector(
            plan_img_or_path=str(self.plan_path),
            trail_len=60, trail_ttl=30, line_thickness=4, point_radius=10,
        )
        logging.info("PlanProjector 초기화 완료")

        # (2) 카메라별 SCST 인스턴스
        self._build_scst_instances()

        logging.info("=== 멀티카메라 비디오 시스템 초기화 완료 ===")
        logging.info(f"도면: {self.plan_path}")
        logging.info(f"카메라 수: {len(self.video_paths)}")
        logging.info(f"출력: {self.output_dir}")

    def _build_scst_instances(self) -> None:
        self.scst_instances.clear()
        for idx in range(len(self.video_paths)):
            scst = videoSCST(
                args=self.args,
                plan_img_path=str(self.plan_path),
                plan_pts=self.plan_points_per_camera[idx],  # 카메라별 플랜 포인트
                det_models=self.detector_models,
            )
            self.scst_instances.append(scst)
        logging.info(f"SCST 인스턴스 생성: {len(self.scst_instances)}개")

    # ───────────────────────────────────────────────────────────────────
    # 실행
    # ───────────────────────────────────────────────────────────────────
    def run(self, parallel: bool = True, max_workers: int = 3) -> bool:
        self.start_time = time.time()
        logging.info("=== 멀티카메라 비디오 처리 시작 ===")

        ok = self._process_parallel(max_workers) if parallel else self._process_sequential()
        if not ok:
            return False

        self._create_unified_plan_video()
        self._generate_summary_report()
        self._cleanup()

        logging.info("=== 멀티카메라 비디오 처리 완료 ===")
        logging.info(f"총 처리 시간: {time.time() - self.start_time:.2f}초")
        logging.info(f"결과 저장: {self.output_dir}")
        return True

    # ───────────────────────────────────────────────────────────────────
    # 단일 카메라 처리
    # ───────────────────────────────────────────────────────────────────
    def _process_single_camera(self, scst: videoSCST, video_path: str, cam_idx: int) -> List[Dict[str, Any]]:
        try:
            logging.info(f"[Cam {cam_idx + 1}] 처리 시작: {video_path}")

            cctv_pts  = self.camera_points[cam_idx]
            plan_pts  = self.plan_points_per_camera[cam_idx]
            cam_out   = self.output_dir / f"tracking_result_cam{cam_idx + 1}.mp4"
            plan_out  = self.output_dir / f"plan_result_cam{cam_idx + 1}.mp4"

            results = scst.track_and_save(
                video_path=video_path,
                cctv_pts=cctv_pts,
                plan_pts=plan_pts,                        # 카메라별 플랜 포인트 전달
                plan_img_path=str(self.plan_path),
                camera_save_path=str(cam_out),
                plan_save_path=str(plan_out),
                plan_mode="bottom-center",
                cam_trail_len=30,
            )

            logging.info(f"[Cam {cam_idx + 1}] 처리 완료: {len(results)} 프레임")
            return results

        except Exception as e:
            logging.error(f"[Cam {cam_idx + 1}] 처리 실패: {e}")
            return []

    # ───────────────────────────────────────────────────────────────────
    # 멀티카메라 처리 (병렬/순차)
    # ───────────────────────────────────────────────────────────────────
    def _process_parallel(self, max_workers: int = 3) -> bool:
        try:
            logging.info(f"병렬 처리 시작 (워커 {max_workers})")
            self.results.clear()

            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                future_to_cam = {
                    exe.submit(self._process_single_camera, scst, vpath, i): i
                    for i, (scst, vpath) in enumerate(zip(self.scst_instances, self.video_paths))
                }
                collected: Dict[int, List[Dict[str, Any]]] = {}
                for future in as_completed(future_to_cam):
                    idx = future_to_cam[future]
                    try:
                        collected[idx] = future.result()
                        logging.info(f"[Cam {idx + 1}] 결과 수집: {len(collected[idx])} 프레임")
                    except Exception as e:
                        logging.error(f"[Cam {idx + 1}] 결과 수집 실패: {e}")
                        collected[idx] = []

                # 카메라 인덱스 정렬
                self.results = [collected[i] for i in range(len(self.video_paths))]

            total_frames = sum(len(r) for r in self.results)
            logging.info(f"모든 카메라 처리 완료: 총 {total_frames} 프레임")
            return True

        except Exception as e:
            logging.error(f"병렬 처리 실패: {e}")
            return False

    def _process_sequential(self) -> bool:
        try:
            logging.info("순차 처리 시작")
            self.results.clear()
            for i, (scst, vpath) in enumerate(zip(self.scst_instances, self.video_paths)):
                self.results.append(self._process_single_camera(scst, vpath, i))

            total_frames = sum(len(r) for r in self.results)
            logging.info(f"모든 카메라 처리 완료: 총 {total_frames} 프레임")
            return True

        except Exception as e:
            logging.error(f"순차 처리 실패: {e}")
            return False

    # ───────────────────────────────────────────────────────────────────
    # 후처리
    # ───────────────────────────────────────────────────────────────────
    def _create_unified_plan_video(self) -> bool:
        """
        주의:
          save_video가 결과 안의 '투영된 경로'만 사용한다면 OK.
          실시간 투영(H 필요) 방식이라면, 카메라별 H가 달라 개별 투영 결과를 합치는 게 안전함.
        """
        try:
            logging.info("통합 플랜 비디오 생성 중...")
            all_results: List[Dict[str, Any]] = []
            for cam_results in self.results:
                all_results.extend(cam_results)

            if not all_results:
                logging.warning("통합할 결과가 없습니다.")
                return False

            unified_output = self.output_dir / "unified_plan_result.mp4"
            self.projector.save_video(all_results, str(unified_output), fps=30.0, mode="bottom-center")
            logging.info(f"통합 플랜 비디오 생성 완료: {unified_output}")
            return True

        except Exception as e:
            logging.error(f"통합 플랜 비디오 생성 실패: {e}")
            return False

        # (필요 시: 개별 plan_result_cam*.mp4들을 타임라인 합성하는 별도 로직을 넣어도 됨)

    def _generate_summary_report(self) -> Dict[str, Any]:
        try:
            total_frames = sum(len(r) for r in self.results)
            total_detections = 0
            total_tracks = 0
            for cam_results in self.results:
                for frame_res in cam_results:
                    if isinstance(frame_res, list):
                        total_detections += len(frame_res)
                        total_tracks += sum(
                            1 for r in frame_res
                            if isinstance(r, dict) and r.get("id") is not None
                        )

            report = {
                "total_cameras": len(self.video_paths),
                "total_frames": total_frames,
                "total_detections": total_detections,
                "total_tracks": total_tracks,
                "output_directory": str(self.output_dir),
                "processing_time": time.time() - getattr(self, "start_time", time.time()),
                "results_per_camera": [len(r) for r in self.results],
            }
            report_file = self.output_dir / "processing_report.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logging.info(f"리포트 저장 완료: {report_file}")
            return report

        except Exception as e:
            logging.error(f"리포트 생성 실패: {e}")
            return {}

    def _cleanup(self) -> None:
        try:
            logging.info("리소스 정리 중...")
            for scst in self.scst_instances:
                if hasattr(scst, "close"):
                    try:
                        scst.close()
                    except Exception:
                        pass
            logging.info("리소스 정리 완료")
        except Exception as e:
            logging.error(f"리소스 정리 중 오류: {e}")


# ───────────────────────────────────────────────────────────────────────────────
# (참고) 사용 예시 — 별도의 main 스크립트에서
# ───────────────────────────────────────────────────────────────────────────────
"""
from MCMT_engine.MCMT_video import MCMTVideoSystem, VideoTrackingArgs

# 공통 플랜 이미지
PLAN_IMG = "/workspace/assets/seocho/Seocho_plan_pts.png"

# 카메라별 플랜 포인트(주석/줄바꿈 유지)
PLAN_POINTS_PER_CAMERA = [cam1_plan_pts, cam2_plan_pts, cam3_plan_pts]

# 카메라별 CCTV 포인트(주석/줄바꿈 유지)
CAMERA_POINTS = [cam1_pts, cam2_pts, cam3_pts]

VIDEO_PATHS = [
    "/workspace/datasets/250909_Site_seocho/2025-09-09 13_29_59 이동형 #1.mp4",
    "/workspace/datasets/250909_Site_seocho/2025-09-09 13_29_59 이동형 #2.mp4",
    "/workspace/datasets/250909_Site_seocho/2025-09-09 13_29_59 이동형 #3.mp4",
]

system = MCMTVideoSystem(
    args=VideoTrackingArgs(),
    plan_path=PLAN_IMG,
    plan_points_per_camera=PLAN_POINTS_PER_CAMERA,
    video_paths=VIDEO_PATHS,
    camera_points=CAMERA_POINTS,
    output_dir="/workspace/results/multi_camera_video",
    detector_models=["ultra_people", "worker"],
)

# 노트북/디버거라면 multiprocessing 이슈 피하려면:
# system.args.cpu_workers = 1

system.run(parallel=True, max_workers=3)
"""
