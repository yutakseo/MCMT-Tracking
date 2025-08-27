from mmdet.apis.inference import init_detector
from mmdet.apis import inference_detector

class WorkerDetector:
    def __init__(self):
        self.model = init_detector(
            config="/workspace/PretrainedModel_by_JeonYT/worker/yolov8x_signalman.py", 
            checkpoint="/workspace/PretrainedModel_by_JeonYT/worker/epoch_100.pth", 
            device="cuda:1"
        )

        self.class_names = [
            'signalman',     
            'worker'    
        ]

        self.id2coco = {
            0: 2,
            1: 3
        }
        self.coco2id = {v: k for k, v in self.id2coco.items()}
        self.coco2name = {v: self.class_names[k] for k, v in self.id2coco.items()}

    def detect(self, image):
        return inference_detector(self.model, image)
