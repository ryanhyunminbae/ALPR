"""
OCR utilities for reading license plate text.

The module wraps EasyOCR with lightweight post-processing to clean and
normalize plate text.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List, Sequence, Tuple

import cv2
import easyocr
import numpy as np

# Characters typically allowed on license plates (alphanumeric ASCII)
ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Mappings for common OCR confusions.
_CHAR_REPLACEMENTS = {
    "O": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
}


def normalize_plate_text(text: str) -> str:
    """
    Clean up OCR output for license plates.

    Steps:
    - Strip whitespace and punctuation
    - Upper-case letters
    - Replace commonly confused characters with more likely alternatives
    - Filter to allowed alphanumeric characters
    """
    if not text:
        return ""

    candidate = re.sub(r"[^A-Za-z0-9]", "", text.upper())

    normalized_chars: List[str] = []
    for char in candidate:
        replacement = _CHAR_REPLACEMENTS.get(char, char)
        if replacement in ALLOWED_CHARS:
            normalized_chars.append(replacement)

    normalized = "".join(normalized_chars)
    return normalized


@lru_cache(maxsize=4)
def _build_reader(languages: Tuple[str, ...], gpu: bool) -> easyocr.Reader:
    """
    Create and cache an EasyOCR reader instance.
    """
    return easyocr.Reader(list(languages), gpu=gpu)


class PlateOCR:
    """
    OCR helper using EasyOCR.

    The reader is cached across instances to avoid the heavy initialization
    cost EasyOCR incurs on first use.
    """

    def __init__(self, languages: Sequence[str] | None = None, gpu: bool = False) -> None:
        self.languages = tuple(languages or ("en",))
        self.gpu = gpu
        self._reader = _build_reader(self.languages, gpu)

    def read_plate(self, image: np.ndarray, candidates: int = 1) -> List[Tuple[str, float]]:
        """
        Run OCR on a license plate crop.

        Parameters
        ----------
        image:
            Plate image (BGR).
        candidates:
            Maximum number of normalized text predictions to return.

        Returns
        -------
        A list of ``(text, confidence)`` tuples in descending confidence order.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result = self._reader.readtext(grayscale, detail=1, allowlist=ALLOWED_CHARS)

        predictions: List[Tuple[str, float]] = []
        for _bbox, raw_text, confidence in sorted(result, key=lambda r: r[2], reverse=True):
            normalized = normalize_plate_text(raw_text)
            if normalized:
                predictions.append((normalized, float(confidence)))
        # Remove duplicates while preserving order
        seen = set()
        unique_predictions: List[Tuple[str, float]] = []
        for text, confidence in predictions:
            if text in seen:
                continue
            seen.add(text)
            unique_predictions.append((text, confidence))
            if len(unique_predictions) >= candidates:
                break
        return unique_predictions
