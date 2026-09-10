"""
Improved ANPR plate + vehicle detector.

Important:
- If a dedicated plate YOLO model exists, it is used first.
- If the model is missing or does not contain a plate class, vehicle detection
  plus a license-plate candidate search is used as a fallback.
- The fallback is deliberately conservative and generates several candidate
  regions instead of assuming the entire lower 45% of a vehicle is a plate.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from config_loader import get_config


@dataclass
class Detection:
    bbox: tuple
    confidence: float
    crop: np.ndarray


class PlateDetector:

    def __init__(
        self,
        model_path: str | None = None,
        conf_threshold: float | None = None,
    ):
        cfg = get_config()["detection"]

        self.model_path = model_path or cfg["plate_model_path"]
        self.conf_threshold = (
            conf_threshold
            if conf_threshold is not None
            else cfg.get("confidence_threshold", 0.35)
        )

        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            if not Path(self.model_path).exists():
                return None

            self._model = YOLO(self.model_path)

        return self._model

    @staticmethod
    def _clip_bbox(bbox, width, height):
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        return x1, y1, x2, y2

    @staticmethod
    def _candidate_plate_regions(vehicle_crop):
        """
        Search for plate-like bright/edge regions inside a vehicle crop.
        """

        if vehicle_crop is None or vehicle_crop.size == 0:
            return []

        h, w = vehicle_crop.shape[:2]

        if w < 40 or h < 25:
            return []

        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)

        # Improve contrast.
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # Two complementary masks.
        bright = cv2.threshold(
            gray, 150, 255, cv2.THRESH_BINARY
        )[1]

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            5,
        )

        candidates = []

        for mask in (bright, adaptive):

            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            for contour in contours:

                x, y, cw, ch = cv2.boundingRect(contour)

                if cw <= 0 or ch <= 0:
                    continue

                area = cw * ch
                aspect = cw / max(ch, 1)

                # Indian plates are generally wide rectangles.
                if not (2.0 <= aspect <= 8.0):
                    continue

                if cw < max(20, int(w * 0.08)):
                    continue

                if ch < max(6, int(h * 0.025)):
                    continue

                if area > w * h * 0.35:
                    continue

                # Prefer plate-like horizontal regions.
                score = (
                    min(aspect / 4.0, 1.0)
                    + min(cw / max(w * 0.5, 1), 1.0)
                )

                candidates.append(
                    (score, x, y, x + cw, y + ch)
                )

        candidates.sort(reverse=True)

        output = []

        # Return only a few strong candidates.
        for _, x1, y1, x2, y2 in candidates[:5]:

            pad_x = max(3, int((x2 - x1) * 0.15))
            pad_y = max(3, int((y2 - y1) * 0.25))

            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)

            output.append((x1, y1, x2, y2))

        return output

    def _detect_with_plate_model(self, frame):
        model = self._load()

        if model is None:
            return []

        names = getattr(model, "names", {})
        has_plate_class = any(
            "plate" in str(name).lower()
            for name in names.values()
        )

        if not has_plate_class:
            return []

        results = model.predict(
            frame,
            conf=self.conf_threshold,
            imgsz=960,
            verbose=False,
        )

        h, w = frame.shape[:2]
        detections = []

        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])

                class_name = str(
                    names.get(class_id, "")
                ).lower()

                if "plate" not in class_name:
                    continue

                confidence = float(box.conf[0])

                if confidence < self.conf_threshold:
                    continue

                bbox = self._clip_bbox(
                    box.xyxy[0].tolist(),
                    w,
                    h,
                )

                x1, y1, x2, y2 = bbox

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                detections.append(
                    Detection(
                        bbox=bbox,
                        confidence=confidence,
                        crop=crop,
                    )
                )

        return detections

    def _detect_vehicle_fallback(self, frame):
        """
        Uses generic YOLO vehicle detection and searches inside each vehicle
        for plate-like regions.
        """

        cfg = get_config()["detection"]

        vehicle_model_path = cfg.get(
            "vehicle_model_path",
            "yolov8n.pt",
        )

        try:
            from ultralytics import YOLO
        except ImportError:
            return []

        if self._model is None:
            pass

        try:
            vehicle_model = YOLO(vehicle_model_path)
        except Exception:
            return []

        results = vehicle_model.predict(
            frame,
            conf=0.25,
            imgsz=960,
            verbose=False,
        )

        h, w = frame.shape[:2]

        # COCO:
        # 2 car
        # 3 motorcycle
        # 5 bus
        # 7 truck
        vehicle_classes = {
            2,
            3,
            5,
            7,
        }

        detections = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                if class_id not in vehicle_classes:
                    continue

                vehicle_conf = float(box.conf[0])

                vx1, vy1, vx2, vy2 = self._clip_bbox(
                    box.xyxy[0].tolist(),
                    w,
                    h,
                )

                if vx2 <= vx1 or vy2 <= vy1:
                    continue

                vehicle_crop = frame[
                    vy1:vy2,
                    vx1:vx2,
                ]

                candidates = self._candidate_plate_regions(
                    vehicle_crop
                )

                # If no strong candidate was found, use a smaller lower
                # vehicle region as a final fallback.
                if not candidates:

                    vh = vy2 - vy1

                    # Lower 35%, rather than the previous 45%.
                    cy1 = int(vh * 0.60)

                    if cy1 < vh - 5:

                        candidates = [
                            (
                                0,
                                0,
                                cy1,
                                vx2 - vx1,
                                vh,
                            )
                        ]

                for cx1, cy1, cx2, cy2 in candidates:

                    px1 = vx1 + cx1
                    py1 = vy1 + cy1
                    px2 = vx1 + cx2
                    py2 = vy1 + cy2

                    px1, py1, px2, py2 = self._clip_bbox(
                        (px1, py1, px2, py2),
                        w,
                        h,
                    )

                    crop = frame[
                        py1:py2,
                        px1:px2,
                    ]

                    if crop.size == 0:
                        continue

                    # Slightly lower confidence because this is a candidate,
                    # not a true plate-model prediction.
                    confidence = min(
                        0.70,
                        max(0.20, vehicle_conf * 0.85),
                    )

                    detections.append(
                        Detection(
                            bbox=(px1, py1, px2, py2),
                            confidence=confidence,
                            crop=crop,
                        )
                    )

        return detections

    def detect(self, frame):
        """
        Return plate detections.
        """

        detections = self._detect_with_plate_model(frame)

        if detections:
            return detections

        return self._detect_vehicle_fallback(frame)


class VehicleDetector:

    COCO_VEHICLE_CLASSES = {
        2: "car",
        3: "motorbike",
        5: "bus",
        7: "truck",
    }

    def __init__(self, model_path: str | None = None):

        cfg = get_config()["detection"]

        self.model_path = (
            model_path
            or cfg.get(
                "vehicle_model_path",
                "yolov8n.pt",
            )
        )

        self._model = None

    def _load(self):

        if self._model is None:

            from ultralytics import YOLO

            self._model = YOLO(self.model_path)

        return self._model

    def detect(self, frame):

        model = self._load()

        results = model.predict(
            frame,
            conf=0.25,
            imgsz=960,
            verbose=False,
        )

        vehicles = []

        for result in results:

            for box in result.boxes:

                cls_id = int(box.cls[0])

                if cls_id not in self.COCO_VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist(),
                )

                vehicles.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "type": self.COCO_VEHICLE_CLASSES[cls_id],
                        "confidence": float(box.conf[0]),
                    }
                )

        return vehicles