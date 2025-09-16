import cv2, os
from typing import List, Tuple, Optional, Literal

Point = Tuple[int, int]
Space = Literal["auto", "orig", "image"]

def _normalize_save_path(image_path: str, save_path: Optional[str]) -> str:
    """
    - save_path가 None이면 <image_stem>_marked.png 로 저장
    - save_path가 디렉터리면 그 안에 <image_stem>_marked.png 로 저장
    - 확장자가 없으면 .png로 보정
    - 상위 폴더 없으면 생성
    """
    img_dir  = os.path.dirname(image_path)
    img_stem = os.path.splitext(os.path.basename(image_path))[0]

    if not save_path:
        save_path = os.path.join(img_dir, f"{img_stem}_marked.png")

    # 디렉터리만 들어온 경우
    if os.path.isdir(save_path) or save_path.endswith(("\\", "/")):
        save_path = os.path.join(save_path, f"{img_stem}_marked.png")

    # 확장자 없으면 .png
    _, ext = os.path.splitext(save_path)
    if not ext:
        save_path += ".png"

    # 상위 폴더 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return save_path

def auto_project_points(
    image_path: str,
    points: List[Point],
    save_path: Optional[str] = None,
    points_space: Space = "auto",
    orig_size: Optional[Tuple[int, int]] = None,
    base_radius: int = 8,
    draw_index: bool = True,
) -> List[Point]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 불러올 수 없습니다: {image_path}")
    h, w = img.shape[:2]

    def _points_fit_image(ps: List[Point]) -> bool:
        for x, y in ps:
            if x < 0 or y < 0 or x > w or y > h:
                return False
        return True

    sx = sy = 1.0
    if points_space == "image":
        scaled_points = [(int(round(x)), int(round(y))) for x, y in points]
    elif points_space == "orig":
        if not orig_size:
            raise ValueError("points_space='orig'이면 orig_size=(W,H)를 지정하세요.")
        ow, oh = orig_size
        sx, sy = w / float(ow), h / float(oh)
        scaled_points = [(int(round(x * sx)), int(round(y * sy))) for x, y in points]
    else:  # auto
        if _points_fit_image(points):
            scaled_points = [(int(round(x)), int(round(y))) for x, y in points]
        else:
            ow = oh = None
            if orig_size:
                ow, oh = orig_size
                sx, sy = w / float(ow), h / float(oh)
            else:
                max_x = max(p[0] for p in points)
                max_y = max(p[1] for p in points)
                sx = (w - 1) / float(max_x)
                sy = (h - 1) / float(max_y)
            scaled_points = [(int(round(x * sx)), int(round(y * sy))) for x, y in points]

    # 표시 크기
    geom_scale = max(w, h) / 2000.0
    radius = max(2, int(base_radius * geom_scale))
    font_scale = max(0.4, 0.6 * geom_scale)

    for idx, (px, py) in enumerate(scaled_points, start=1):
        cv2.circle(img, (px, py), radius + 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (px, py), radius, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        if draw_index:
            text = str(idx)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            tx, ty = px - tw // 2, py - radius - 4
            cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)

    # 저장 경로 정규화 + 저장
    save_path = _normalize_save_path(image_path, save_path)
    ok = cv2.imwrite(save_path, img)
    if not ok:
        raise RuntimeError(
            "이미지 저장 실패: cv2.imwrite가 False를 반환했습니다.\n"
            f"- save_path: {save_path}\n"
            "- 원인 후보: 쓰기 권한/디스크 가득 참/경로 이상 문자"
        )

    print(f"✅ 저장: {save_path} | 입력 {w}x{h} | pts={len(points)} | radius={radius} | "
          f"mode={points_space} | sx={sx:.4f} sy={sy:.4f}")
    return scaled_points
