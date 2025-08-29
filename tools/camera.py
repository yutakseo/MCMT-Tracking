from __Detection.detection_api import DetectionAPI
from __Tracking.tracking_api import TrackerAPI
from tools.homo_graphy import PlanProjector
import cv2

class Args:
    track_thresh = 0.5
    match_thresh = 0.5
    track_buffer = 60
    mot20 = False
    cpu_workers = 10  
    chunk_sec   = 10.0

class Camera:
    def __init__(self, cctv_url:str, cord_plan:str, calibration:list,):
        self.url = cctv_url
        self.plan_path = cord_plan
        self.calibration = calibration
        
        self.detector = DetectionAPI()
        self.tracker = TrackerAPI(args=Args(), detector=self.detector)
        return None

    def webcam_show(self, show:bool):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("Webcam", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):  # q 누르면 종료
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return None
    
    def stream(self, show:bool):
        
        return None