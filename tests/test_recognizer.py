
import numpy as np
import pytest
from unittest.mock import patch

from src.face_recognizer import FaceRecognizer


@pytest.fixture
def recognizer():
    with (
        patch("src.face_recognizer.FaceDetector"),
        patch("src.face_recognizer.FaceEmbedder"),
        patch("src.face_recognizer.FaceMatcher"),
    ):
        yield FaceRecognizer()


def test_recognize_before_register(recognizer):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(RuntimeError):
        recognizer.recognize(frame)


def test_register_one_face(recognizer):
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    recognizer.detector.detect.return_value = [
        {"box": [10, 10, 50, 50]}
    ]

    recognizer.embedder.extract.return_value = np.ones(
        128, dtype=np.float32
    )

    with patch(
        "src.face_recognizer.align_face",
        return_value=image
    ):
        recognizer.register(image)

    assert recognizer.reference_embedding.shape == (128,)


def test_register_multiple_faces(recognizer):
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    recognizer.detector.detect.return_value = [
        {"box": [10, 10, 30, 30]},
        {"box": [50, 10, 30, 30]}
    ]

    with pytest.raises(ValueError):
        recognizer.register(image)


def test_recognize_matched_face(recognizer):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    recognizer.reference_embedding = np.ones(
        128, dtype=np.float32
    )

    recognizer.detector.detect.return_value = [
        {"box": [10, 20, 50, 60]}
    ]

    recognizer.embedder.extract.return_value = np.ones(
        128, dtype=np.float32
    )

    recognizer.matcher.match.return_value = {
        "similarity": 0.85,
        "label": "Matched"
    }

    with patch(
        "src.face_recognizer.align_face",
        return_value=frame
    ):
        results = recognizer.recognize(frame)

    assert len(results) == 1
    assert results[0]["label"] == "Matched"
    assert results[0]["similarity"] == 0.85


def test_recognize_no_faces(recognizer):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    recognizer.reference_embedding = np.ones(
        128, dtype=np.float32
    )

    recognizer.detector.detect.return_value = []

    results = recognizer.recognize(frame)

    assert results == []
