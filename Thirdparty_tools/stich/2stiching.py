"""
VSCode에서 바로 실행되는 1x2(가로) 영상 병합 스크립트.
- 서로 다른 해상도/FPS/길이 지원
- 기준 세로(height)로 리사이즈, 가로는 비율 유지
- 가장 긴/짧은 영상 기준 선택
- 부족 구간은 마지막 프레임 유지(last) 또는 검정 화면(black) 패딩
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass

# =========================
# CONFIG: 여기만 바꿔주세요
# =========================
INPUT_LEFT   = r"/workspace/results/tracking_result3new.mp4"   # 왼쪽 영상
INPUT_RIGHT  = r"/workspace/results/plan_result3.mp4"  # 오른쪽 영상
OUTPUT       = r"/workspace/results/cam#3.mp4"

TARGET_HEIGHT = 1080          # 출력 세로(px). 0이면 입력 중 가장 작은 세로
MODE = "longest"             # "longest": 가장 긴 영상 끝까지, "shortest": 가장 짧은 영상에 맞춰 종료
PAD = "last"                 # "last": 마지막 프레임 유지, "black": 검정화면
OUTPUT_FPS = 0.0             # 0이면 입력 FPS들의 "최솟값" 사용, 0보다 크면 고정 FPS
FOURCC = "mp4v"              # "mp4v" or "H264"(환경에 따라 필요 코덱), 확장자 .avi 사용 가능

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
    planned_w: int = 0
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


def resize_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if target_h <= 0 or h == target_h:
        return img
    new_w = int(round(w * (target_h / float(h))))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_LINEAR)


def make_black(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def plan_output_and_warmup(vids: list, target_h: int):
    """첫 프레임 리사이즈로 폭을 확정 → hstack 안정화(프레임마다 폭 변동 방지)"""
    heights = [v.h for v in vids if v.h > 0]
    if target_h <= 0:
        target_h = min(heights) if heights else 720

    for v in vids:
        ok, f = v.cap.read()
        if not ok or f is None:
            base_h = v.h if v.h > 0 else target_h
            base_w = v.w if v.w > 0 else target_h * 16 // 9
            f = make_black(base_h, base_w)
        fr = resize_to_height(f, target_h)
        v.last_resized = fr.copy()
        v.planned_w = fr.shape[1]
        v.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_w = sum(v.planned_w for v in vids)
    return target_h, total_w


def decide_output_fps(vids: list, user_fps: float) -> float:
    if user_fps and user_fps > 0:
        return float(user_fps)
    fps_pos = [v.fps for v in vids if v.fps and v.fps > 0]
    return float(min(fps_pos) if fps_pos else 30.0)


def open_writer(out_path: str, size_hw: tuple, fps: float, fourcc_str: str) -> cv2.VideoWriter:
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (size_hw[1], size_hw[0]))
    if not writer.isOpened():
        raise RuntimeError(
            "VideoWriter를 열 수 없습니다. 다른 코덱(FOURCC='H264') 또는 출력 확장자(.avi)로 시도해 보세요."
        )
    return writer


def safe_read_and_resize(v: Vid, target_h: int, pad_mode: str) -> np.ndarray:
    if v.finished:
        return v.last_resized if pad_mode == "last" else make_black(target_h, v.planned_w)

    ok, f = v.cap.read()
    if not ok or f is None:
        v.finished = True
        return v.last_resized if pad_mode == "last" else make_black(target_h, v.planned_w)

    fr = resize_to_height(f, target_h)
    # 계획된 폭과 다르면 패딩/크롭으로 고정 폭 유지
    if fr.shape[1] != v.planned_w:
        if fr.shape[1] < v.planned_w:
            pad = v.planned_w - fr.shape[1]
            fr = cv2.copyMakeBorder(fr, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        else:
            fr = fr[:, :v.planned_w]
    v.last_resized = fr
    return fr


def main():
    vids = [read_meta(INPUT_LEFT), read_meta(INPUT_RIGHT)]
    print("[INFO] 입력 영상:")
    for i, v in enumerate(vids):
        print(f"  {i}: {os.path.basename(v.path)} | {v.w}x{v.h} | {v.fps:.2f} fps | frames={v.count}")

    out_h, total_w = plan_output_and_warmup(vids, TARGET_HEIGHT)
    out_size = (out_h, total_w)  # (H, W)

    out_fps = decide_output_fps(vids, OUTPUT_FPS)
    print(f"[INFO] 출력 크기: {out_size[1]}x{out_size[0]}, FPS={out_fps:.2f}, MODE={MODE}, PAD={PAD}, FOURCC={FOURCC}")

    writer = open_writer(OUTPUT, out_size, out_fps, FOURCC)

    step = 0
    try:
        while True:
            left  = safe_read_and_resize(vids[0], out_h, PAD)
            right = safe_read_and_resize(vids[1], out_h, PAD)

            if MODE == "shortest":
                if vids[0].finished or vids[1].finished:
                    break
            else:  # longest
                if vids[0].finished and vids[1].finished:
                    break

            out_frame = cv2.hconcat([left, right])
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
        print(f"\n[OK] 저장 완료: {OUTPUT}  size={out_size[1]}x{out_size[0]}  fps={out_fps:.2f}")


if __name__ == "__main__":
    main()
