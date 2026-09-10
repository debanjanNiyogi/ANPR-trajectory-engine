"""
Standalone ANPR video tester.

Usage:

    python scripts/test_anpr_video.py --video sample_data/cam1.mp4

Optional:

    python scripts/test_anpr_video.py \
        --video sample_data/cam1.mp4 \
        --output output/result.mp4 \
        --csv output/detections.csv \
        --display

The script:
- reads the video
- detects vehicles / plate candidates
- performs OCR
- draws detections
- saves an annotated video
- saves CSV logs
- prints detection information
"""

import argparse
import csv
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2


# Allow imports when running:
# python scripts/test_anpr_video.py
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from detection.plate_detector import (
    PlateDetector,
    VehicleDetector,
)

from detection.ocr_reader import (
    OCRReader,
    validate_plate,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Test ANPR detection on a video."
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Input video path",
    )

    parser.add_argument(
        "--output",
        default="output/anpr_result.mp4",
        help="Annotated output video",
    )

    parser.add_argument(
        "--csv",
        default="output/detections.csv",
        help="CSV detection log",
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Show live OpenCV window",
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum plate confidence",
    )

    return parser.parse_args()


def ensure_parent(path):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def draw_label(
    frame,
    text,
    x,
    y,
):

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(
        frame,
        (
            x,
            max(0, y - 28),
        ),
        (
            x + max(180, len(text) * 10),
            y,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x + 5, y - 7),
        font,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():

    args = parse_args()

    video_path = Path(
        args.video
    )

    if not video_path.exists():

        print(
            f"ERROR: video not found: "
            f"{video_path}"
        )

        sys.exit(1)

    ensure_parent(args.output)
    ensure_parent(args.csv)

    print()
    print("=" * 60)
    print("ANPR VIDEO TEST")
    print("=" * 60)
    print(
        f"Input : {video_path}"
    )
    print(
        f"Output: {args.output}"
    )
    print(
        f"CSV   : {args.csv}"
    )
    print("=" * 60)
    print()

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():

        print(
            "ERROR: OpenCV could not open video."
        )

        sys.exit(1)

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 25.0

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    print(
        f"Video: {width}x{height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    print(
        f"Frames: {total_frames}"
    )

    output_path = Path(
        args.output
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():

        print(
            "ERROR: could not create "
            "output video."
        )

        cap.release()
        sys.exit(1)

    print()
    print(
        "Loading YOLO models..."
    )

    plate_detector = PlateDetector(
        conf_threshold=args.confidence
    )

    vehicle_detector = (
        VehicleDetector()
    )

    print(
        "Loading OCR..."
    )

    ocr_reader = OCRReader(
        conf_threshold=0.20
    )

    print(
        "Models loaded."
    )

    print()

    csv_file = open(
        args.csv,
        "w",
        newline="",
        encoding="utf-8",
    )

    csv_writer = csv.writer(
        csv_file
    )

    csv_writer.writerow(
        [
            "frame",
            "time_seconds",
            "plate",
            "det_confidence",
            "ocr_confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
    )

    frame_index = 0

    processed_frames = 0

    vehicle_count = 0

    candidate_count = 0

    ocr_count = 0

    valid_count = 0

    plate_counter = Counter()

    first_seen = {}

    last_print = time.time()

    started = time.time()

    try:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1

            if (
                args.frame_skip > 1
                and frame_index
                % args.frame_skip
                != 0
            ):
                writer.write(frame)
                continue

            processed_frames += 1

            # Vehicle detection.
            try:

                vehicles = (
                    vehicle_detector.detect(
                        frame
                    )
                )

                vehicle_count += len(
                    vehicles
                )

            except Exception as exc:

                print(
                    f"Vehicle detector error: "
                    f"{exc}"
                )

                vehicles = []

            # Plate detection.
            try:

                plate_dets = (
                    plate_detector.detect(
                        frame
                    )
                )

            except Exception as exc:

                print(
                    f"Plate detector error: "
                    f"{exc}"
                )

                plate_dets = []

            candidate_count += len(
                plate_dets
            )

            # Draw vehicle boxes.
            for vehicle in vehicles:

                x1, y1, x2, y2 = (
                    vehicle["bbox"]
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 180, 0),
                    2,
                )

                draw_label(
                    frame,
                    (
                        f'{vehicle["type"]} '
                        f'{vehicle["confidence"]:.2f}'
                    ),
                    x1,
                    y1,
                )

            # OCR each plate candidate.
            for det in plate_dets:

                x1, y1, x2, y2 = (
                    det.bbox
                )

                # Candidate box.
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                result = None

                try:

                    result = (
                        ocr_reader.read(
                            det.crop
                        )
                    )

                except Exception as exc:

                    print(
                        f"OCR error: {exc}"
                    )

                if result is None:

                    draw_label(
                        frame,
                        (
                            f"PLATE ? "
                            f"{det.confidence:.2f}"
                        ),
                        x1,
                        y1,
                    )

                    continue

                ocr_count += 1

                plate_text, ocr_conf = (
                    result
                )

                if not validate_plate(
                    plate_text
                ):

                    draw_label(
                        frame,
                        (
                            f"{plate_text} "
                            f"{ocr_conf:.2f}"
                        ),
                        x1,
                        y1,
                    )

                    continue

                valid_count += 1

                plate_counter[
                    plate_text
                ] += 1

                first_seen.setdefault(
                    plate_text,
                    frame_index,
                )

                seconds = (
                    frame_index / fps
                )

                csv_writer.writerow(
                    [
                        frame_index,
                        f"{seconds:.2f}",
                        plate_text,
                        f"{det.confidence:.4f}",
                        f"{ocr_conf:.4f}",
                        x1,
                        y1,
                        x2,
                        y2,
                    ]
                )

                print(
                    f"[FRAME {frame_index:06d}] "
                    f"PLATE={plate_text} "
                    f"det={det.confidence:.2f} "
                    f"ocr={ocr_conf:.2f}"
                )

                draw_label(
                    frame,
                    (
                        f"{plate_text} "
                        f"{ocr_conf:.2f}"
                    ),
                    x1,
                    y1,
                )

            # Information overlay.
            elapsed = (
                time.time() - started
            )

            processing_fps = (
                processed_frames
                / max(elapsed, 0.001)
            )

            overlay = (
                f"Frames: {frame_index} | "
                f"Vehicles: {vehicle_count} | "
                f"Plate candidates: "
                f"{candidate_count} | "
                f"OCR: {ocr_count} | "
                f"Valid: {valid_count} | "
                f"FPS: {processing_fps:.1f}"
            )

            cv2.rectangle(
                frame,
                (0, 0),
                (width, 35),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                overlay,
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            writer.write(frame)

            if args.display:

                cv2.imshow(
                    "ANPR Test",
                    frame,
                )

                key = cv2.waitKey(1)

                if key & 0xFF == ord(
                    "q"
                ):
                    print(
                        "Stopped by user."
                    )
                    break

            # Periodic progress.
            if (
                time.time()
                - last_print
                > 5
            ):

                print(
                    f"Progress: "
                    f"{frame_index}/"
                    f"{total_frames} | "
                    f"vehicles="
                    f"{vehicle_count} | "
                    f"candidates="
                    f"{candidate_count} | "
                    f"valid="
                    f"{valid_count}"
                )

                last_print = time.time()

    finally:

        cap.release()
        writer.release()
        csv_file.close()

        if args.display:
            cv2.destroyAllWindows()

    elapsed = (
        time.time() - started
    )

    print()
    print("=" * 60)
    print("ANPR TEST COMPLETE")
    print("=" * 60)

    print(
        f"Frames processed : "
        f"{processed_frames}"
    )

    print(
        f"Vehicles detected: "
        f"{vehicle_count}"
    )

    print(
        f"Plate candidates : "
        f"{candidate_count}"
    )

    print(
        f"OCR successes    : "
        f"{ocr_count}"
    )

    print(
        f"Valid plate reads: "
        f"{valid_count}"
    )

    print(
        f"Unique plates    : "
        f"{len(plate_counter)}"
    )

    print(
        f"Processing time  : "
        f"{elapsed:.1f}s"
    )

    print()

    if plate_counter:

        print(
            "Recognized plates:"
        )

        for plate, count in (
            plate_counter.most_common()
        ):

            print(
                f"  {plate:<15} "
                f"{count} frames"
            )

    else:

        print(
            "WARNING: No valid plates "
            "were recognized."
        )

        print(
            "Check the annotated output "
            "video to determine whether "
            "the problem is detection "
            "or OCR."
        )

    print()
    print(
        f"Annotated video: "
        f"{output_path}"
    )

    print(
        f"CSV log: {args.csv}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()