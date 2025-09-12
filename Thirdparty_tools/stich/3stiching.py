"""
VSCode에서 바로 실행되는 3x1(세로) 영상 병합 스크립트.
- 서로 다른 해상도/FPS/길이 지원
- 기준 가로(width)로 리사이즈, 세로는 비율 유지
- 가장 긴/짧은 영상 기준 동작 선택
- 부족 구간은 마지막 프레임 유지(last) 또는 검정(black) 패딩
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass

# =========================
# CONFIG: 여기만 바꿔주세요
# =========================
INPUTS = [
    r"D:\OneDrive - 스마트인사이드에이아이\문서\32.프로젝트\__2023_02_STEAM_한국연구재단(2023.04. - 2027.12.)\12_workspace\SIAI-MCMT_homography solution\250903_실내실험2_성균관대2종합연구동\results\tracking_result1.mp4",  # 위(Top)
    r"D:\OneDrive - 스마트인사이드에이아이\문서\32.프로젝트\__2023_02_STEAM_한국연구재단(2023.04. - 2027.12.)\12_workspace\SIAI-MCMT_homography solution\250903_실내실험2_성균관대2종합연구동\results\tracking_result2.mp4",  # 중간(Middle)
    r"D:\OneDrive - 스마트인사이드에이아이\문서\32.프로젝트\__2023_02_STEAM_한국연구재단(2023.04. - 2027.12.)\12_workspace\SIAI-MCMT_homography solution\250903_실내실험2_성균관대2종합연구동\results\tracking_result3.mp4",  # 아래(Bottom)
]
OUTPUT = r"D:\videos\merged_out_vertical.mp4"

TARGET_WIDTH = 1280         # 출력 가로(px). 0이면 입력 중 가장 작은 가로
MODE = "longest"            # "longest": 가장 긴 영상 끝까지, "shortest": 가장 짧은 영상에 맞춰 종료
PAD = "last"                # "last": 마지막 프레임 유지, "black": 검정화면
OUTPUT_FPS = 0.0            # 0이면 입력 FPS들의 "최솟값" 사용, 0보다 크면 고정 FPS
FOURCC = "mp4v"             # "mp4v" or "H264"(환경에 따라 필요 코덱), 확장자 .avi도 가능

# (RTSP에 주로 영향) 지연/끊김 완화 옵션
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000|buffer_size;102400"


@dataclass
class Vid:
    path: str
    cap: cv2.VideoCapture
    fps: float
    w: int
    h: int
    count: int
    last_resized: np.ndarray = None
    planned_h: int = 0
    finished: bool = False


def read_meta(path: str) -> Vid:
    if not os.path.exists(path):
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return Vid(path, cap, fps, w, h, cnt)


def resize_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if target_w <= 0 or w == target_w:
        return img
    new_h = int(round(h * (target_w / float(w))))
    return cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_LINEAR)


def make_black(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def plan_output_and_warmup(vids: list, target_w: int):
    """첫 프레임으로 리사이즈 높이를 확정 → vstack 안정화(프레임마다 높이 변동 방지)"""
    widths = [v.w for v in vids if v.w > 0]
    if target_w <= 0:
        target_w = min(widths) if widths else 1280

    for v in vids:
        ok, f = v.cap.read()
        if not ok or f is None:
            # 첫 프레임이 없다면 임시 검정 프레임
            base_w = v.w if v.w > 0 else target_w
            base_h = v.h if v.h > 0 else target_w * 9 // 16
            f = make_black(base_h, base_w)
        fr = resize_to_width(f, target_w)
        v.last_resized = fr.copy()
        v.planned_h = fr.shape[0]
        v.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 다시 처음으로
    total_h = sum(v.planned_h for v in vids)
    return target_w, total_h


def decide_output_fps(vids: list, user_fps: float) -> float:
    if user_fps and user_fps > 0:
        return float(user_fps)
    fps_pos = [v.fps for v in vids if v.fps and v.fps > 0]
    return float(min(fps_pos) if fps_pos else 30.0)


def open_writer(out_path: str, size_wh: tuple, fps: float, fourcc_str: str) -> cv2.VideoWriter:
    # size_wh = (W, H)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(out_path, fourcc, fps, size_wh)
    if not writer.isOpened():
        raise RuntimeError("VideoWriter를 열 수 없습니다. FOURCC='H264' 또는 확장자 .avi로 시도해 보세요.")
    return writer


def safe_read_and_resize(v: Vid, target_w: int, pad_mode: str) -> np.ndarray:
    if v.finished:
        return v.last_resized if pad_mode == "last" else make_black(v.planned_h, target_w)

    ok, f = v.cap.read()
    if not ok or f is None:
        v.finished = True
        return v.last_resized if pad_mode == "last" else make_black(v.planned_h, target_w)

    fr = resize_to_width(f, target_w)
    # 계획된 높이와 다르면 패딩/크롭으로 고정 높이 유지 (가로는 target_w로 고정)
    if fr.shape[0] != v.planned_h:
        if fr.shape[0] < v.planned_h:
            pad = v.planned_h - fr.shape[0]
            fr = cv2.copyMakeBorder(fr, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            fr = fr[:v.planned_h, :]

    # 혹시 가로가 미세하게 어긋나면 보정
    if fr.shape[1] != target_w:
        if fr.shape[1] < target_w:
            pad = target_w - fr.shape[1]
            fr = cv2.copyMakeBorder(fr, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            fr = fr[:, :target_w]

    v.last_resized = fr
    return fr


def main():
    if len(INPUTS) != 3:
        raise ValueError("INPUTS에는 정확히 3개의 영상 경로를 넣어주세요. (위→중간→아래 순서)")

    # 1) 메타 읽기
    vids = [read_meta(p) for p in INPUTS]
    print("[INFO] 입력 영상:")
    for i, v in enumerate(vids):
        print(f"  {i}: {os.path.basename(v.path)} | {v.w}x{v.h} | {v.fps:.2f} fps | frames={v.count}")

    # 2) 출력 가로/세로 확정 (첫 프레임으로 각 높이 계획)
    out_w, total_h = plan_output_and_warmup(vids, TARGET_WIDTH)
    out_size_wh = (out_w, total_h)  # (W, H)

    # 3) 출력 FPS 결정
    out_fps = decide_output_fps(vids, OUTPUT_FPS)
    print(f"[INFO] 출력 크기: {out_size_wh[0]}x{out_size_wh[1]}, FPS={out_fps:.2f}, MODE={MODE}, PAD={PAD}, FOURCC={FOURCC}")

    # 4) VideoWriter 준비
    writer = open_writer(OUTPUT, out_size_wh, out_fps, FOURCC)

    # 5) 루프
    step = 0
    try:
        while True:
            tiles = []
            for v in vids:
                tiles.append(safe_read_and_resize(v, out_w, PAD))

            if MODE == "shortest":
                if any(v.finished for v in vids):
                    break
            else:  # longest
                if all(v.finished for v in vids):
                    break

            out_frame = cv2.vconcat(tiles)
            writer.write(out_frame)
            step += 1
            if step % int(max(out_fps, 1)) == 0:
                print(f"[INFO] 작성 프레임: {step}", end="\r")

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단")

    finally:
        for v in vids:
            v.cap.release()
        writer.release()
        print(f"\n[OK] 저장 완료: {OUTPUT}  size={out_size_wh[0]}x{out_size_wh[1]}  fps={out_fps:.2f}")


if __name__ == "__main__":
    main()