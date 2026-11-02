
import numpy as np
import pytest
from unittest.mock import patch

from src.face_detector import FaceDetector


def test_detect_empty_frame():
    with patch("src.face_detector.MTCNN"):
        detector = FaceDetector()

    with pytest.raises(ValueError):
        detector.detect(None)


def test_detect_blank_image():
    with patch("src.face_detector.MTCNN") as mock_mtcnn:
        mock_mtcnn.return_value.detect_faces.return_value = []

        detector = FaceDetector()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = detector.detect(frame)

        assert faces == []


def test_detect_one_face():
    with patch("src.face_detector.MTCNN") as mock_mtcnn:
        mock_mtcnn.return_value.detect_faces.return_value = [
            {
                "box": [10, 20, 50, 60],
                "confidence": 0.99
            }
        ]

        detector = FaceDetector()

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        faces = detector.detect(frame)

        assert len(faces) == 1
        assert faces[0]["box"] == [10, 20, 50, 60]
