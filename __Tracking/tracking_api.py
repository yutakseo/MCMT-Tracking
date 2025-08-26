from __Tracking.ByteTrack.yolox.tracker.byte_tracker import BYTETracker
import cv2, os
import numpy as np
from hashlib import md5
from collections import defaultdict, deque  # ✅ 추가

class TrackerAPI:
    def __init__(self, args, detector) -> None:
        """
        args: tracker 설정값 객체 (track_thresh, match_thresh, track_buffer 등)
        detector: .detect(image) -> (N,5 or N,6) 텐서/넘파이 반환 (x1,y1,x2,y2,score,(cls))
        """
        self.tracker = BYTETracker(args)
        self.detector = detector

    def tracking(self, video_path, visualize=False):
        """
        video_path: 비디오 파일 경로 (ex: "video.mp4")
        visualize: True일 경우 결과를 화면에 그려줌
        """
        results = []
        img_size = None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # 첫 프레임에서 img_size 자동 결정
            if img_size is None:
                fh, fw = frame.shape[:2]
                img_size = (fh, fw)

            # 1) Detection
            dets = self.detector.detect(frame)

            # 2) Tracking
            fh, fw = frame.shape[:2]
            info_imgs = (fh, fw)
            online_targets = self.tracker.update(dets, info_imgs, img_size)

            frame_res = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                score = t.score
                frame_res.append({
                    "id": tid,
                    "bbox": tlwh,   # (x, y, w, h) float32
                    "score": score
                })

                if visualize:
                    x1, y1, bw, bh = map(int, tlwh)  # ✅ 변수명 충돌 방지
                    cv2.rectangle(frame, (x1, y1), (x1 + bw, y1 + bh), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID:{tid}", (x1, max(0, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            results.append(frame_res)

            if visualize:
                cv2.imshow("tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if visualize:
            cv2.destroyAllWindows()
        return results

    def _color_from_id(self, track_id: int):
        # ID마다 고정 색상
        h = md5(str(track_id).encode()).hexdigest()
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))  # BGR-ish

    def trackingVideo(self, video_path, save_path, trail_len: int = 30):
        """
        video_path: 입력 영상 경로
        save_path : 저장할 결과 영상 경로
        trail_len: 궤적 길이(저장할 최근 중심점 개수)
        """
        # 1) detection+tracking 실행 (화면표시는 하지 않음)
        results = self.tracking(video_path=video_path, visualize=False)

        # 2) 비디오 입출력 준비
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {video_path}")

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps) or fps <= 1e-3:
            fps = 30.0  # 기본값

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            alt_path = os.path.splitext(save_path)[0] + ".avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(alt_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError("VideoWriter open failed for both mp4v and MJPG.")
            save_path = alt_path

        # 3) ID별 궤적 저장소: tid -> deque([(x,y), ...], maxlen=trail_len)
        trails = defaultdict(lambda: deque(maxlen=trail_len))
        trail_thickness = 2  # ✅ 고정값

        # 4) 프레임 순회하며 시각화 + 저장
        for frame_id, frame_res in enumerate(results):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            current_ids = set()
            for t in frame_res:
                x, y, bw, bh = map(int, t["bbox"])
                tid = int(t["id"])
                score = float(t["score"])
                current_ids.add(tid)

                color = self._color_from_id(tid)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(frame, f"ID:{tid} {score:.2f}",
                            (x, max(0, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # 중심점 갱신
                cx = x + bw * 0.5
                cy = y + bh * 0.5
                trails[tid].append((cx, cy))

            # 궤적 그리기 (현재 프레임에 존재하는 ID만)
            for tid in current_ids:
                pts = trails[tid]
                if len(pts) >= 2:
                    pts_np = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts_np], isClosed=False,
                                  color=self._color_from_id(tid), thickness=trail_thickness)
                if len(pts) >= 1:
                    cx, cy = map(int, pts[-1])
                    cv2.circle(frame, (cx, cy), radius=2 + trail_thickness,
                               color=self._color_from_id(tid), thickness=-1)

            writer.write(frame)

        cap.release()
        writer.release()
        print(f"결과 영상 저장 완료: {save_path}")
        return results
