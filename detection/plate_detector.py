"""
Wraps a YOLOv8 model for license-plate localization within a video frame.

Swap `plate_model_path` in config.yaml for your own fine-tuned weights —
generic COCO-trained YOLO doesn't have a "license_plate" class, so for real
accuracy you need a model trained on a plate dataset (e.g. OpenALPR/CCPD/your
own annotated city footage). This wrapper is model-agnostic: any YOLOv8
.pt file whose class 0 is "plate" will work unmodified.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from config_loader import get_config


@dataclass
class Detection:
    bbox: tuple  # (x1, y1, x2, y2) in pixel coords
    confidence: float
    crop: np.ndarray  # cropped plate region, ready for OCR


class PlateDetector:
    def __init__(self, model_path: str | None = None, conf_threshold: float | None = None):
        cfg = get_config()["detection"]
        self.model_path = model_path or cfg["plate_model_path"]
        self.conf_threshold = conf_threshold or cfg["confidence_threshold"]
        self._model = None  # lazy-loaded (keeps import fast / testable without weights)

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def _bright_plate_regions(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 225, 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            aspect_ratio = width / max(height, 1)
            if not (30 <= width <= 180 and 6 <= height <= 30 and 2.5 <= aspect_ratio <= 18):
                continue
            x1, y1 = max(0, x - 3), max(0, y - 3)
            x2, y2 = min(frame.shape[1], x + width + 3), min(frame.shape[0], y + height + 3)
            detections.append(Detection(
                bbox=(x1, y1, x2, y2), confidence=1.0,
                crop=frame[y1:y2, x1:x2],
            ))
        return detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run plate detection on a single BGR frame, return crops ready for OCR."""
        model = self._load()
        results = model.predict(frame, conf=self.conf_threshold, verbose=False)
        has_plate_class = any("plate" in str(name).lower() for name in model.names.values())

        detections = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                if not has_plate_class:
                    if class_id not in {2, 3, 5, 7}:
                        continue
                    vehicle_x1, vehicle_y1, vehicle_x2, vehicle_y2 = map(int, box.xyxy[0].tolist())
                    vehicle_height = vehicle_y2 - vehicle_y1
                    x1, y1, x2, y2 = (
                        vehicle_x1, vehicle_y1 + int(vehicle_height * 0.55),
                        vehicle_x2, vehicle_y2,
                    )
                else:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=conf, crop=crop))
            return detections or self._bright_plate_regions(frame) if not has_plate_class else detections


class VehicleDetector:
    """
    Optional secondary detector (generic COCO YOLO) used to classify vehicle
    type (car/truck/bus/motorbike) for the vehicle_type column and to give
    the tracker a bounding box to follow between frames even when the plate
    itself is briefly unreadable (glare, angle, occlusion).
    """
    COCO_VEHICLE_CLASSES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

    def __init__(self, model_path: str | None = None):
        cfg = get_config()["detection"]
        self.model_path = model_path or cfg["vehicle_model_path"]
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame: np.ndarray) -> list[dict]:
        model = self._load()
        results = model.predict(frame, verbose=False)
        vehicles = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.COCO_VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                vehicles.append({
                    "bbox": (x1, y1, x2, y2),
                    "type": self.COCO_VEHICLE_CLASSES[cls_id],
                    "confidence": float(box.conf[0]),
                })
        return vehicles
