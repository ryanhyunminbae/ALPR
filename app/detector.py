"""
License plate detection utilities powered by YOLO.

The detector is fed with a YOLOv8 model that has been trained offline on a
license-plate dataset. The exported weights live under ``app/models`` and are
loaded at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass(frozen=True)
class Detection:
    """Represents a single detected license plate."""

    bbox: Tuple[int, int, int, int]
    confidence: float = 1.0

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Return the region of the image enclosed by the bounding box."""
        x, y, w, h = self.bbox
        return image[y : y + h, x : x + w]


class PlateDetector:
    """
    License plate detector backed by a YOLOv8 model.

    Parameters
    ----------
    weights_path:
        Optional custom path to the YOLO weights file. Defaults to the bundled
        demo model trained on license plates.
    confidence:
        Confidence threshold for YOLO predictions.
    """

    def __init__(
        self,
        weights_path: Path | str | None = None,
        confidence: float = 0.25,
        iou: float = 0.45,
    ) -> None:
        self.confidence = confidence
        self.iou = iou
        self._weights_path = Path(weights_path) if weights_path else self._default_weights_path()

        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found at {self._weights_path}. "
                "Train or download the detector weights and place them under app/models."
            )

        try:
            self._model = YOLO(str(self._weights_path))
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to load YOLO weights from {self._weights_path}") from exc

    @staticmethod
    def _default_weights_path() -> Path:
        """Return the default location of the YOLO weights file."""
        return Path(__file__).resolve().parent / "models" / "yolov8n-plates.pt"

    def detect(
        self,
        image: np.ndarray,
    ) -> List[Detection]:
        """
        Detect license plates in an image.

        Parameters
        ----------
        image:
            Input image as a NumPy array in BGR color order.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        # YOLO expects RGB input.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb_image.shape[:2]

        results = self._model.predict(
            rgb_image,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        detections: List[Detection] = []

        if not results:
            return detections

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Clamp coordinates to image bounds and convert to integers.
                x1_i = max(0, min(int(round(x1)), width - 1))
                y1_i = max(0, min(int(round(y1)), height - 1))
                x2_i = max(0, min(int(round(x2)), width))
                y2_i = max(0, min(int(round(y2)), height))

                w = max(0, x2_i - x1_i)
                h = max(0, y2_i - y1_i)
                if w == 0 or h == 0:
                    continue

                confidence = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
                detections.append(Detection(bbox=(x1_i, y1_i, w, h), confidence=confidence))

        return detections

    @staticmethod
    def draw_detections(image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
        """
        Draw bounding boxes for detections onto a copy of the image.
        """
        overlay = image.copy()
        for detection in detections:
            x, y, w, h = detection.bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{detection.confidence:.2f}"
            cv2.putText(
                overlay,
                label,
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return overlay


def load_image(image_path: Path | str) -> np.ndarray:
    """
    Read an image from disk in BGR format.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image at {path}")
    return image

