"""
Robust OCR reader for ANPR.

Runs several preprocessing variants and chooses the strongest OCR result.
"""

import re

import cv2
import numpy as np

from config_loader import get_config


PLATE_TEXT_REGEX = re.compile(
    r"^[A-Z0-9]{4,12}$"
)


def clean_plate_text(raw: str) -> str:
    text = re.sub(
        r"[^A-Za-z0-9]",
        "",
        raw or "",
    )

    return text.upper()


def validate_plate(text: str) -> bool:

    if not text:
        return False

    if not PLATE_TEXT_REGEX.fullmatch(text):
        return False

    # Avoid accepting obvious OCR garbage.
    if len(set(text)) == 1:
        return False

    return 4 <= len(text) <= 12


def _resize_crop(crop):

    h, w = crop.shape[:2]

    if h <= 0 or w <= 0:
        return crop

    # Plates need to be sufficiently large for OCR.
    target_height = 160

    scale = max(
        2.0,
        target_height / float(h),
    )

    scale = min(scale, 8.0)

    return cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def _preprocess_variants(crop):

    enlarged = _resize_crop(crop)

    gray = cv2.cvtColor(
        enlarged,
        cv2.COLOR_BGR2GRAY,
    )

    # Variant 1: CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    enhanced = clahe.apply(gray)

    # Variant 2: OTSU
    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    # Variant 3: adaptive
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )

    # Variant 4: sharpened grayscale
    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        3,
    )

    sharpened = cv2.addWeighted(
        gray,
        1.5,
        blur,
        -0.5,
        0,
    )

    # Variant 5: inverted OTSU
    inverted = cv2.bitwise_not(otsu)

    return [
        enlarged,
        enhanced,
        otsu,
        adaptive,
        sharpened,
        inverted,
    ]


class OCRReader:

    def __init__(
        self,
        conf_threshold: float | None = None,
        langs: list[str] | None = None,
        gpu: bool | None = None,
    ):

        cfg = get_config()["detection"]

        self.conf_threshold = (
            conf_threshold
            if conf_threshold is not None
            else cfg.get(
                "ocr_confidence_threshold",
                0.25,
            )
        )

        self.langs = langs or ["en"]

        if gpu is None:
            gpu = cfg.get(
                "ocr_gpu",
                False,
            )

        self.gpu = gpu

        self._reader = None

    def _load(self):

        if self._reader is None:

            import easyocr

            try:

                self._reader = easyocr.Reader(
                    self.langs,
                    gpu=self.gpu,
                    verbose=False,
                )

            except Exception:

                # Automatic CPU fallback.
                self._reader = easyocr.Reader(
                    self.langs,
                    gpu=False,
                    verbose=False,
                )

        return self._reader

    def read(self, crop):

        if crop is None or crop.size == 0:
            return None

        reader = self._load()

        variants = _preprocess_variants(crop)

        candidates = []

        for image in variants:

            try:

                results = reader.readtext(
                    image,
                    allowlist=(
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        "0123456789"
                    ),
                    paragraph=False,
                    detail=1,
                    mag_ratio=1.0,
                )

            except Exception:
                continue

            if not results:
                continue

            fragments = []

            confidences = []

            for result in results:

                if len(result) < 3:
                    continue

                text = clean_plate_text(
                    result[1]
                )

                conf = float(result[2])

                if not text:
                    continue

                fragments.append(text)
                confidences.append(conf)

            if not fragments:
                continue

            combined = "".join(fragments)

            avg_conf = (
                sum(confidences)
                / len(confidences)
            )

            if len(combined) >= 4:

                # Reward longer plausible strings.
                score = (
                    avg_conf
                    + min(len(combined), 10) * 0.015
                )

                candidates.append(
                    (
                        score,
                        combined,
                        avg_conf,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        _, text, confidence = candidates[0]

        if confidence < self.conf_threshold:
            return None

        if not validate_plate(text):
            return None

        return text, confidence