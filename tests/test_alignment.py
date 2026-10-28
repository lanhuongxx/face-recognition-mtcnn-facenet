
import numpy as np
import pytest

from src.face_alignment import align_face


def make_face():
    return {
        "box": [30, 30, 60, 60],
        "keypoints": {
            "left_eye": (45, 50),
            "right_eye": (75, 50)
        }
    }


def test_align_valid_face():
    image = np.zeros((120, 120, 3), dtype=np.uint8)

    aligned = align_face(image, make_face())

    assert aligned.ndim == 3
    assert aligned.shape[2] == 3
    assert aligned.size > 0


def test_align_empty_image():
    with pytest.raises(ValueError):
        align_face(None, make_face())


def test_align_tilted_eyes():
    image = np.zeros((120, 120, 3), dtype=np.uint8)

    face = make_face()
    face["keypoints"]["right_eye"] = (75, 65)

    aligned = align_face(image, face)

    assert aligned.size > 0
    assert aligned.shape[2] == 3


def test_align_box_outside_image():
    image = np.zeros((120, 120, 3), dtype=np.uint8)

    face = make_face()
    face["box"] = [200, 200, 60, 60]

    with pytest.raises(ValueError):
        align_face(image, face)
