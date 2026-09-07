"""
Reads plate text out of a cropped plate image using EasyOCR, with basic
preprocessing, cleanup, and region-neutral plate validation.
"""
import re
import cv2
import numpy as np

from config_loader import get_config

PLATE_TEXT_REGEX = re.compile(r"^[A-Z0-9]{4,12}$")


def _preprocess(crop: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.equalizeHist(gray)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def clean_plate_text(raw: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    return text


def validate_plate(text: str) -> bool:
    """Accept common plate text across regions without enforcing a format."""
    return bool(PLATE_TEXT_REGEX.fullmatch(text))


class OCRReader:
    def __init__(self, conf_threshold: float | None = None, langs: list[str] | None = None):
        cfg = get_config()["detection"]
        self.conf_threshold = conf_threshold or cfg["ocr_confidence_threshold"]
        self.langs = langs or ["en"]
        self._reader = None

    def _load(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(self.langs, gpu=True)
        return self._reader

    def read(self, crop: np.ndarray) -> tuple[str, float] | None:
        """Returns (plate_text, confidence) or None if nothing readable/valid."""
        if crop is None or crop.size == 0:
            return None
        reader = self._load()
        processed = _preprocess(crop)
        results = reader.readtext(
            processed,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            paragraph=False,
        )
        if not results:
            return None

        # Concatenate all detected text fragments (plates sometimes split into
        # two OCR boxes, e.g. state code line + number line), weighted by conf.
        combined_text = "".join(r[1] for r in results)
        avg_conf = sum(r[2] for r in results) / len(results)

        text = clean_plate_text(combined_text)
        if avg_conf < self.conf_threshold or len(text) < 6:
            return None

        return text, avg_conf
