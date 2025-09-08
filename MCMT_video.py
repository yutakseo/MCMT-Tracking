# /workspace/MCMT_video.py
# Multi-Camera Multi-Target Video Processing System (refactored)

"""
멀티카메라 비디오 처리 시스템
- 카메라별 비디오를 독립 추적 후, 결과(카메라 영상/도면 투영)를 저장
- 모든 카메라의 플랜(도면) 결과를 통합해 단일 비디오 생성
- 처리 요약 리포트 저장
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# 외부(프로젝트 내부) 의존성
import sys
sys.path.append("/workspace")
from __Detection.detection_api import DetectionAPI
from __Tracking.tracking_api import TrackerAPI
from MCMT_engine.streaming.video_SCST import videoSCST, Args
from tools.homo_graphy import PlanProjector


# ───────────────────────────────────────────────────────────────────────────────
# 기본 상수 (필요 시 외부에서 주입 가능)
# ───────────────────────────────────────────────────────────────────────────────
PLAN_PATH = "/workspace/assets/250904_homograph_coordinate-plane2.jpg"
PLAN_POINTS: List[Tuple[float, float]] = [
    (1170, 214), (1170, 559), (1170, 904),
    (2212, 214), (2212, 559), (2212, 904),
    (3255, 214), (3255, 559), (3255, 904),
]
CAMERA_POINTS: List[List[Tuple[float, float]]] = [
    [(1033, 475), (948, 474), (863, 473), (1019, 527), (890, 524), (769, 519), (973, 667), (741, 652), (548, 620)],  # Cam1
    [(518, 466), (430, 471), (341, 474), (613, 510), (485, 519), (354, 527), (829, 608), (634, 645), (397, 668)],   # Cam2
    [(357, 602), (566, 648), (832, 683), (620, 498), (754, 509), (893, 513), (726, 453), (819, 456), (911, 459)],   # Cam3
]


# ───────────────────────────────────────────────────────────────────────────────
# 추적 파라미터
# ───────────────────────────────────────────────────────────────────────────────
class VideoTrackingArgs(Args):
    """비디오 추적 시스템 설정(필요시 적절히 조정)"""
    track_thresh = 0.3
    match_thresh = 0.9
    track_buffer = 180
    mot20 = False
    cpu_workers = 20
    chunk_sec = 10.0
    batch_size = 20


# ───────────────────────────────────────────────────────────────────────────────
# 메인 시스템 클래스
# ───────────────────────────────────────────────────────────────────────────────
class MCMTVideoSystem:
    """
    멀티카메라 비디오 처리 시스템(정리/리팩터링 버전)

    사용 순서:
      1) create_video_tracking_system(...) 으로 인스턴스 생성(모델/프로젝터/SCST 준비)
      2) run(parallel=True/False) 실행
    """

    def __init__(
        self,
        args: VideoTrackingArgs,
        plan_path: str,
        plan_points: List[Tuple[float, float]],
        video_paths: List[str],
        camera_points: List[List[Tuple[float, float]]],
        output_dir: str = "/workspace/results",
        detector_models: Optional[List[str]] = None,
    ):
        # 기본 설정/검증
        if len(video_paths) != len(camera_points):
            raise ValueError(
                f"비디오 수({len(video_paths)})와 카메라 포인트 수({len(camera_points)})가 일치하지 않습니다."
            )

        self.args = args
        self.plan_path: Path = Path(plan_path)
        self.plan_points: List[Tuple[float, float]] = plan_points
        self.video_paths: List[str] = video_paths
        self.camera_points: List[List[Tuple[float, float]]] = camera_points
        self.output_dir: Path = Path(output_dir)
        self.detector_models: Optional[List[str]] = detector_models

        # 출력 디렉토리 준비
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 결과/인스턴스 컨테이너
        self.results: List[List[Dict[str, Any]]] = []  # 카메라별 프레임 결과 모음
        self.scst_instances: List[videoSCST] = []

        # 로깅 기본 설정
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
        """공유 Detector/Tracker/Projector 및 카메라별 SCST 인스턴스 초기화"""
        logging.info("모델/프로젝터 초기화 중...")

        # 1) Detector
        self.shared_detector = DetectionAPI(
            models=self.detector_models,
            thres=0.0,
            device="cuda:0",
            use_async=True,
            max_workers=1,
        )
        logging.info(f"Detector 초기화 완료: models={self.detector_models}")

        # 2) Tracker
        self.shared_tracker = TrackerAPI(args=self.args, detector=self.shared_detector)
        logging.info("Tracker 초기화 완료")

        # 3) Homography Projector (공유)
        self.projector = PlanProjector(
            plan_img_or_path=str(self.plan_path),
            trail_len=60,
            trail_ttl=30,
            line_thickness=4,
            point_radius=10,
        )
        logging.info("PlanProjector 초기화 완료")

        # GPU 메모리 상태 간단 표기
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                logging.info(f"GPU 메모리: {allocated:.2f} GB")
        except Exception:
            pass

        # 4) 카메라별 SCST 인스턴스 생성
        self._build_scst_instances()

        logging.info("=== 멀티카메라 비디오 시스템 초기화 완료 ===")
        logging.info(f"도면: {self.plan_path}")
        logging.info(f"도면 포인트: {len(self.plan_points)}개")
        logging.info(f"비디오: {len(self.video_paths)}개")
        logging.info(f"카메라 포인트: {len(self.camera_points)}개")
        logging.info(f"출력: {self.output_dir}")

    def _build_scst_instances(self) -> None:
        """
        카메라 수만큼 videoSCST 인스턴스를 생성한다.
        - 라이브러리 시그니처 차이를 고려해 몇 가지 생성 방식을 순차 시도한다.
        """
        self.scst_instances.clear()
        for idx in range(len(self.video_paths)):
            scst = self._make_scst_safe()
            self.scst_instances.append(scst)
        logging.info(f"SCST 인스턴스 생성: {len(self.scst_instances)}개")

    def _make_scst_safe(self) -> videoSCST:
        """
        videoSCST 생성 시 여러 시그니처를 안전하게 시도.
        프로젝트 환경에 맞춰 한 가지는 반드시 성공하도록 구성.
        """
        # 시그니처 ①
        try:
            return videoSCST(
                args=self.args,
                detector=self.shared_detector,
                tracker=self.shared_tracker,
                projector=self.projector,
            )
        except Exception:
            pass

        # 시그니처 ②
        try:
            return videoSCST(self.args, self.shared_detector, self.shared_tracker, self.projector)
        except Exception:
            pass

        # 시그니처 ③ (가장 단순)
        try:
            return videoSCST(self.args)
        except Exception as e:
            raise RuntimeError(f"videoSCST 인스턴스 생성 실패: {e}") from e

    # ───────────────────────────────────────────────────────────────────
    # 실행 엔트리
    # ───────────────────────────────────────────────────────────────────
    def run(self, parallel: bool = True, max_workers: int = 3) -> bool:
        """전체 처리 파이프라인 실행"""
        self.start_time = time.time()
        logging.info("=== 멀티카메라 비디오 처리 시작 ===")

        ok = self._process_parallel(max_workers) if parallel else self._process_sequential()
        if not ok:
            return False

        # 통합 플랜 비디오 + 요약 리포트
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
        """
        카메라 한 대의 비디오를 처리하고,
        카메라 영상/플랜 투영 비디오를 저장하며,
        프레임별 결과(list[dict])를 반환한다.
        """
        try:
            logging.info(f"[Cam {cam_idx + 1}] 처리 시작: {video_path}")

            # 카메라별 호모그래피 포인트
            cctv_pts = self.camera_points[cam_idx]

            # 출력 경로
            camera_output = self.output_dir / f"tracking_result_cam{cam_idx + 1}.mp4"
            plan_output = self.output_dir / f"plan_result_cam{cam_idx + 1}.mp4"

            # 비디오 처리 및 추적
            results = scst.track_and_save(
                video_path=video_path,
                cctv_pts=cctv_pts,
                camera_save_path=str(camera_output),
                plan_save_path=str(plan_output),
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

                for future in as_completed(future_to_cam):
                    cam_idx = future_to_cam[future]
                    try:
                        cam_results = future.result()
                        self.results.append(cam_results)
                        logging.info(f"[Cam {cam_idx + 1}] 결과 수집: {len(cam_results)} 프레임")
                    except Exception as e:
                        logging.error(f"[Cam {cam_idx + 1}] 결과 수집 실패: {e}")
                        self.results.append([])

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
                cam_results = self._process_single_camera(scst, vpath, i)
                self.results.append(cam_results)

            total_frames = sum(len(r) for r in self.results)
            logging.info(f"모든 카메라 처리 완료: 총 {total_frames} 프레임")
            return True

        except Exception as e:
            logging.error(f"순차 처리 실패: {e}")
            return False

    # ───────────────────────────────────────────────────────────────────
    # 후처리: 통합 플랜 비디오 / 요약 리포트 / 정리
    # ───────────────────────────────────────────────────────────────────
    def _create_unified_plan_video(self) -> bool:
        """모든 카메라의 플랜 좌표를 모아 단일 비디오로 저장"""
        try:
            logging.info("통합 플랜 비디오 생성 중...")
            all_results: List[Dict[str, Any]] = []
            for cam_results in self.results:
                all_results.extend(cam_results)

            if not all_results:
                logging.warning("통합할 결과가 없습니다.")
                return False

            unified_output = self.output_dir / "unified_plan_result.mp4"

            # 공유 projector 사용(첫 SCST projector 대신)
            self.projector.save_video(
                all_results,
                str(unified_output),
                fps=30.0,
                mode="bottom-center",
            )

            logging.info(f"통합 플랜 비디오 생성 완료: {unified_output}")
            return True

        except Exception as e:
            logging.error(f"통합 플랜 비디오 생성 실패: {e}")
            return False

    def _generate_summary_report(self) -> Dict[str, Any]:
        """처리 결과 요약 리포트를 JSON으로 저장"""
        try:
            total_frames = sum(len(r) for r in self.results)
            total_detections = 0
            total_tracks = 0

            # 결과 포맷에 따라 집계 로직을 조정하세요.
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
        """리소스 정리"""
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
# 팩토리 함수 (외부에서 쓰기 쉬운 형태)
# ───────────────────────────────────────────────────────────────────────────────
def create_video_tracking_system(
    args: Optional[VideoTrackingArgs] = None,
    plan_path: str = PLAN_PATH,
    plan_points: List[Tuple[float, float]] = PLAN_POINTS,
    video_paths: Optional[List[str]] = None,
    camera_points: List[List[Tuple[float, float]]] = CAMERA_POINTS,
    detector_models: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    output_dir: str = "/workspace/results/multi_camera_video",
) -> MCMTVideoSystem:
    """
    멀티카메라 비디오 시스템 생성 헬퍼
    - Args/경로/포인트/모델명 등을 받아 MCMTVideoSystem 을 구성
    """
    if args is None:
        args = VideoTrackingArgs()
    if video_paths is None:
        video_paths = []

    system = MCMTVideoSystem(
        args=args,
        plan_path=plan_path,
        plan_points=plan_points,
        video_paths=video_paths,
        camera_points=camera_points,
        output_dir=output_dir,
        detector_models=detector_models,
        class_names=class_names,
    )
    return system


# ───────────────────────────────────────────────────────────────────────────────
# 메인 실행 예시
# ───────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 예시: 3개 카메라 비디오 처리
    video_paths = [
        "/workspace/datasets/homography_experiment2/1100-1110/2025-09-04 10_59_59 이동형 #1_part_000.mp4",
        "/workspace/datasets/homography_experiment2/1100-1110/2025-09-04 10_59_58 이동형 #2_part_000.mp4",
        "/workspace/datasets/homography_experiment2/1100-1110/2025-09-04 10_59_58 이동형 #3_part_000.mp4",
    ]

    system = create_video_tracking_system(
        args=VideoTrackingArgs(),
        plan_path=PLAN_PATH,
        plan_points=PLAN_POINTS,
        video_paths=video_paths,
        camera_points=CAMERA_POINTS,
        detector_models=["ultra_people", "worker"],
        class_names=["People"],
        output_dir="/workspace/results/multi_camera_video",
    )

    logging.info("✅ 멀티카메라 비디오 시스템 생성 완료")
    logging.info(f"Detector: {system.shared_detector}")
    logging.info(f"Tracker : {system.shared_tracker}")
    logging.info(f"Projector: {system.projector}")

    # 실행
    system.run(parallel=True, max_workers=3)
