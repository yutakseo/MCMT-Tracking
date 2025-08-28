from mmdet.apis.inference import init_detector
from mmdet.apis import inference_detector
from _registry_ import register_detector



@register_detector("vehicle")
class VehicleDetector:
    def __init__(self):
        self.model = init_detector(
            config="/workspace/PretrainedModel_by_JeonYT/vehicle/yolov8x_vehicle.py", 
            checkpoint="/workspace/PretrainedModel_by_JeonYT/vehicle/epoch_54.pth", 
            device="cuda:0"
        )

        self.class_names = [
            'dump_truck',    
            'excavator',     
            'forklift',       
            'mixer_truck',    
            'scissor_lift',   
            'bulldozer',      
            'cargo_truck',    
            'crane'           
        ]

        self.id2coco = {
            0: 9,
            1: 10,
            2: 11,
            3: 12,
            4: 13,
            5: 16,
            6: 17,
            7: 18
        }
        self.coco2id = {v: k for k, v in self.id2coco.items()}
        self.coco2name = {v: self.class_names[k] for k, v in self.id2coco.items()}
        
    def detect(self, image):
        return inference_detector(self.model, image)
