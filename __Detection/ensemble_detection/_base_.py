#_base_.py
from .vehicle import VehicleDetector
from .worker import WorkerDetector

class EnsembleDetector:
    def __init__(self, thres=0.3):
        self.vehicle = VehicleDetector()
        self.worker = WorkerDetector()
        self.thres = thres
        
    def _detect(self, image):
        vehicle_results = self.vehicle.detect(image)
        worker_results = self.worker.detect(image)
        return vehicle_results, worker_results

    def _parse_detsample(self, sub_result, id2coco=None, coco2name=None):
        parsed = []
        preds = sub_result.pred_instances

        for i in range(len(preds.labels)):
            score = preds.scores[i].item()
            if score < self.thres:
                continue
            label_id = preds.labels[i].item()
            coco_id = id2coco.get(label_id, label_id) if id2coco else label_id
            label = coco2name.get(coco_id, f"unknown_{coco_id}") if coco2name else f"label_{coco_id}"
            bbox = preds.bboxes[i].tolist()
            parsed.append({
                "class_id": coco_id,
                "label": label,
                "score": score,
                "bbox": bbox
            })
        return parsed



    def detect(self, image):
        _detected_vehicles, _detected_workers = self._detect(image)

        vehicle = self._parse_detsample(
            _detected_vehicles,
            id2coco=self.vehicle.id2coco,
            coco2name=self.vehicle.coco2name
        )

        worker = self._parse_detsample(
            _detected_workers,
            id2coco=self.worker.id2coco,
            coco2name=self.worker.coco2name
        )

        return vehicle + worker
