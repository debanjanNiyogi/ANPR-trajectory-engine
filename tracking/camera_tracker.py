import logging
import time
from datetime import datetime, timezone

import cv2

from config_loader import get_camera, get_config
from detection.plate_detector import (
    PlateDetector,
    VehicleDetector,
)
from detection.ocr_reader import (
    OCRReader,
    validate_plate,
)
from db import database
from alerting import alert_engine


logger = logging.getLogger(__name__)


DEDUP_WINDOW_SECONDS = 8.0


class CameraTracker:

    def __init__(self, camera_id: str):

        self.camera_id = camera_id

        self.camera_cfg = get_camera(
            camera_id
        )

        detection_cfg = get_config()[
            "detection"
        ]

        self.frame_skip = int(
            detection_cfg.get(
                "frame_skip",
                1,
            )
        )

        self.plate_detector = PlateDetector()

        self.vehicle_detector = (
            VehicleDetector()
        )

        self.ocr_reader = OCRReader()

        self._recent_plates = {}

        self.stats = {
            "frames": 0,
            "plate_candidates": 0,
            "ocr_success": 0,
            "valid_plates": 0,
            "logged": 0,
        }

    def _already_seen_recently(
        self,
        plate,
        now,
    ):

        last_seen = self._recent_plates.get(
            plate
        )

        if (
            last_seen is not None
            and now - last_seen
            < DEDUP_WINDOW_SECONDS
        ):
            return True

        self._recent_plates[plate] = now

        return False

    def _match_vehicle_type(
        self,
        plate_bbox,
        vehicles,
    ):

        px1, py1, px2, py2 = plate_bbox

        best = None
        best_area = 0

        for vehicle in vehicles:

            vx1, vy1, vx2, vy2 = vehicle[
                "bbox"
            ]

            ix1 = max(px1, vx1)
            iy1 = max(py1, vy1)
            ix2 = min(px2, vx2)
            iy2 = min(py2, vy2)

            if ix2 <= ix1 or iy2 <= iy1:
                continue

            intersection = (
                ix2 - ix1
            ) * (
                iy2 - iy1
            )

            if intersection > best_area:

                best_area = intersection
                best = vehicle["type"]

        return best

    def process_frame(
        self,
        frame,
        event_context=None,
    ):

        events = []

        now = time.time()

        event_context = (
            event_context or {}
        )

        self.stats["frames"] += 1

        plate_dets = (
            self.plate_detector.detect(
                frame
            )
        )

        self.stats[
            "plate_candidates"
        ] += len(plate_dets)

        if not plate_dets:
            return events

        try:
            vehicles = (
                self.vehicle_detector.detect(
                    frame
                )
            )
        except Exception:
            vehicles = []

        for det in plate_dets:

            ocr_result = (
                self.ocr_reader.read(
                    det.crop
                )
            )

            if ocr_result is None:
                continue

            self.stats["ocr_success"] += 1

            plate_text, ocr_conf = (
                ocr_result
            )

            if not validate_plate(
                plate_text
            ):
                continue

            self.stats[
                "valid_plates"
            ] += 1

            if self._already_seen_recently(
                plate_text,
                now,
            ):
                continue

            vehicle_type = (
                self._match_vehicle_type(
                    det.bbox,
                    vehicles,
                )
            )

            timestamp = (
                datetime.now(
                    timezone.utc
                ).isoformat()
            )

            database.insert_detection(
                plate_number=plate_text,
                camera_id=self.camera_id,
                timestamp=timestamp,
                confidence=det.confidence,
                ocr_confidence=ocr_conf,
                bbox=det.bbox,
                vehicle_type=vehicle_type,
                signal_state=event_context.get(
                    "signal_state"
                ),
                crossed_stop_line=event_context.get(
                    "crossed_stop_line"
                ),
                travel_direction=event_context.get(
                    "travel_direction"
                ),
                lane_type=event_context.get(
                    "lane_type"
                ),
            )

            self.stats["logged"] += 1

            detection_event = {
                "plate_number": plate_text,
                "camera_id": self.camera_id,
                "timestamp": timestamp,
                "vehicle_type": vehicle_type,
                "confidence": det.confidence,
                "ocr_confidence": ocr_conf,
                "bbox": det.bbox,
            }

            try:
                alert_engine.evaluate_detection(
                    detection_event
                )
            except Exception:
                logger.exception(
                    "Alert engine failed"
                )

            events.append(
                detection_event
            )

            logger.info(
                "[%s] PLATE=%s "
                "det=%.2f ocr=%.2f",
                self.camera_id,
                plate_text,
                det.confidence,
                ocr_conf,
            )

        return events

    def run(
        self,
        source_override=None,
        on_progress=None,
        replay=False,
    ):

        source = (
            source_override
            or self.camera_cfg["source"]
        )

        cap = cv2.VideoCapture(
            source
        )

        if not cap.isOpened():

            logger.error(
                "[%s] Could not open %s",
                self.camera_id,
                source,
            )

            return False

        frame_index = 0

        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_index += 1

                if (
                    self.frame_skip > 1
                    and frame_index
                    % self.frame_skip
                    != 0
                ):
                    continue

                events = (
                    self.process_frame(
                        frame
                    )
                )

                if on_progress:

                    on_progress(
                        frame_index,
                        events,
                        self.stats,
                    )

        finally:

            cap.release()

        logger.info(
            "[%s] Finished: %s",
            self.camera_id,
            self.stats,
        )

        return True