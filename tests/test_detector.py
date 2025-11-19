import numpy as np
import pytest

from app.detector import PlateDetector


def test_detector_handles_blank_image():
    try:
        detector = PlateDetector()
    except FileNotFoundError:
        pytest.skip("YOLO weights not available for detector tests.")

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = detector.detect(image)
    assert isinstance(detections, list)
    for detection in detections:
        assert len(detection.bbox) == 4

