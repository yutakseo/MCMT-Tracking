import subprocess
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

# -------------------------------------------------------
# 핵심: Windows 호환 MP4(H.264 + AAC, yuv420p, faststart)
# -------------------------------------------------------
def convert_to_windows_mp4(
    input_path: str,
    output_path: Optional[str] = None,
    crf: int = 20,
    preset: str = "veryfast",
    fps: Optional[float] = None,             # ex) 30.0 로 지정하면 CFR로 변환
    resolution: Optional[Tuple[int, int]] = None,  # (width, height) 예: (1920,1080)
    audio_bitrate: str = "192k",
    two_pass: bool = False,
    keep_subtitles: bool = True
) -> Path:
    """
    영상을 Windows 친화적인 MP4(H.264 + AAC)로 변환.

    Parameters
    ----------
    input_path : str
        입력 영상 경로
    output_path : Optional[str]
        출력 경로 (None이면 입력파일과 같은 폴더에 *_win.mp4)
    crf : int
        화질 지표 (낮을수록 고화질, 18~23 권장)
    preset : str
        인코딩 속도/압축률 트레이드오프 (ultrafast ~ placebo)
    fps : Optional[float]
        지정 시 CFR(고정 프레임레이트)로 변환. Windows 호환성↑
    resolution : Optional[(int,int)]
        지정 시 리사이즈 (예: (1280,720))
    audio_bitrate : str
        오디오 비트레이트 (예: "128k", "192k")
    two_pass : bool
        True면 2-pass 인코딩(품질/용량 안정↑, 느림)
    keep_subtitles : bool
        자막이 있으면 MP4 호환 mov_text로 변환해 포함

    Returns
    -------
    Path
        변환된 파일 경로
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg가 설치되어 있지 않습니다. (https://ffmpeg.org)")

    in_path = Path(input_path).resolve()
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path = Path(output_path).resolve() if output_path else in_path.with_name(f"{in_path.stem}_win.mp4")

    # 공통 비디오/오디오 옵션 (Windows 호환)
    # -pix_fmt yuv420p : 대부분의 Windows 플레이어 호환
    # -movflags +faststart : 스트리밍/빠른 시작
    # -profile:v high -level 4.1 : 호환성↑
    vopts = [
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level:v", "4.1",
        "-preset", preset,
        "-crf", str(crf),
    ]

    # FPS 고정(원하면) + VFR→CFR 안정화
    # -vsync cfr : CFR 강제
    if fps is not None:
        vopts += ["-vsync", "cfr", "-r", str(fps)]
    else:
        # 기본은 소스 FPS 유지(VFR일 땐 플레이어에 따라 이슈 있을 수 있음)
        vopts += ["-vsync", "vfr"]

    # 해상도 리사이즈(옵션)
    vf_chain: List[str] = []
    if resolution is not None:
        w, h = resolution
        vf_chain.append(f"scale={w}:{h}")

    if vf_chain:
        vopts += ["-vf", ",".join(vf_chain)]

    aopts = [
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-ac", "2",  # 스테레오로 다운믹스 (호환성↑)
    ]

    # 자막 처리: MP4 호환 mov_text로 변환
    # (일부 .ass/.srt를 내장가능. Windows 기본 앱에서 표시 호환성은 앱마다 다름)
    subopts: List[str] = []
    if keep_subtitles:
        subopts = ["-c:s", "mov_text"]
    else:
        subopts = ["-sn"]  # subtitle none

    common_tail = ["-movflags", "+faststart", "-y", str(out_path)]

    def run_ffmpeg(cmd: List[str]):
        print(">>", " ".join(cmd))
        subprocess.run(cmd, check=True)

    if two_pass:
        # 1-pass
        passlog = str(out_path.with_suffix(""))  # ffmpeg는 자동으로 -0.log 같은 파일 생성
        cmd1 = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-i", str(in_path),
                *vopts, "-pass", "1",
                "-an",  # 1pass는 영상만
                "-f", "mp4",  # dummy mux
                "/dev/null" if not is_windows() else "NUL"]
        run_ffmpeg(cmd1)

        # 2-pass
        cmd2 = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
                "-i", str(in_path),
                *vopts, "-pass", "2",
                *aopts, *subopts,
                *common_tail]
        run_ffmpeg(cmd2)
    else:
        # 단일 패스
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
               "-i", str(in_path),
               *vopts, *aopts, *subopts,
               *common_tail]
        run_ffmpeg(cmd)

    print(f"✅ 변환 완료: {out_path}")
    return out_path


def batch_convert_folder(
    folder: str,
    patterns: Tuple[str, ...] = (".mp4", ".mkv", ".mov", ".avi", ".webm"),
    **kwargs
):
    """폴더 안의 모든 영상 파일을 일괄 변환"""
    folder_path = Path(folder).resolve()
    for p in folder_path.iterdir():
        if p.is_file() and p.suffix.lower() in patterns:
            try:
                convert_to_windows_mp4(str(p), **kwargs)
            except Exception as e:
                print(f"⚠️ {p.name} 변환 실패: {e}")


def is_windows() -> bool:
    import platform
    return platform.system().lower().startswith("win")


# -------------------------
# 여기서 바로 실행 예시
# -------------------------
if __name__ == "__main__":
    # 예시 1) 단일 파일 변환: 1080p로 리사이즈, 30fps CFR
    convert_to_windows_mp4(
        input_path=r"C:\Users\서유탁\Downloads\plan_projection2.mp4",
        output_path=r"C:\Users\서유탁\Downloads\plan_projection2_converted.mp4",          # None이면 sample_video_win.mp4
        crf=20,
        preset="veryfast",
        fps=30.0,                  # CFR로 고정: Windows 호환성↑
        resolution=(1920, 1080),   # 리사이즈 (생략 가능)
        audio_bitrate="192k",
        two_pass=False,            # 품질/용량 안정화가 필요하면 True
        keep_subtitles=True
    )

    # 예시 2) 폴더 일괄 변환 (원본 해상도/프레임 유지)
    # batch_convert_folder("./videos", fps=None, resolution=None, crf=21, preset="faster")
