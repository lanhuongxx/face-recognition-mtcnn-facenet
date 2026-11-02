
import cv2
import numpy as np
import pytest

from src.face_tracker import OpticalFlowTracker


def make_frame():
    """Create a synthetic textured face region."""
    rng = np.random.default_rng(42)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    noise = rng.integers(
        0, 256, size=(100, 100, 3), dtype=np.uint8
    )
    frame[60:160, 100:200] = noise

    return frame


def test_initial_state():
    tracker = OpticalFlowTracker()

    assert tracker.is_tracking is False
    assert tracker.box is None
    assert tracker.update(make_frame()) is None


def test_initialize_success():
    tracker = OpticalFlowTracker()
    frame = make_frame()

    result = tracker.initialize(frame, (100, 60, 100, 100))

    assert result is True
    assert tracker.is_tracking is True
    assert tracker.box == (100, 60, 100, 100)
    assert tracker.previous_points is not None
    assert len(tracker.previous_points) >= tracker.MIN_POINTS


@pytest.mark.parametrize(
    "box",
    [
        (100, 60, 0, 100),
        (100, 60, 100, -1),
        (400, 60, 100, 100),
        (100, 300, 100, 100),
    ],
)
def test_initialize_invalid_box(box):
    tracker = OpticalFlowTracker()

    with pytest.raises(ValueError):
        tracker.initialize(make_frame(), box)


def test_initialize_empty_frame():
    tracker = OpticalFlowTracker()

    with pytest.raises(ValueError):
        tracker.initialize(None, (100, 60, 100, 100))


def test_update_same_frame():
    tracker = OpticalFlowTracker()
    frame = make_frame()

    assert tracker.initialize(frame, (100, 60, 100, 100))

    box = tracker.update(frame.copy())

    assert box is not None

    x, y, w, h = box

    assert abs(x - 100) <= 2
    assert abs(y - 60) <= 2
    assert abs(w - 100) <= 2
    assert abs(h - 100) <= 2


def test_update_translation():
    tracker = OpticalFlowTracker()
    frame = make_frame()

    assert tracker.initialize(frame, (100, 60, 100, 100))

    transform = np.float32(
        [
            [1, 0, 10],
            [0, 1, 5],
        ]
    )

    moved = cv2.warpAffine(
        frame,
        transform,
        (320, 240),
    )

    box = tracker.update(moved)

    assert box is not None

    x, y, w, h = box

    assert abs(x - 110) <= 3
    assert abs(y - 65) <= 3
    assert abs(w - 100) <= 3
    assert abs(h - 100) <= 3


def test_update_scale():
    tracker = OpticalFlowTracker()
    frame = make_frame()

    assert tracker.initialize(frame, (100, 60, 100, 100))

    center = (150, 110)

    transform = cv2.getRotationMatrix2D(
        center,
        0,
        1.05,
    )

    scaled = cv2.warpAffine(
        frame,
        transform,
        (320, 240),
    )

    box = tracker.update(scaled)

    assert box is not None

    _, _, w, h = box

    assert w >= 100
    assert h >= 100


def test_reset():
    tracker = OpticalFlowTracker()
    frame = make_frame()

    assert tracker.initialize(frame, (100, 60, 100, 100))

    tracker.reset()

    assert tracker.is_tracking is False
    assert tracker.box is None
    assert tracker.previous_gray is None
    assert tracker.previous_points is None


def test_update_lost_tracking():
    tracker = OpticalFlowTracker()
    frame = make_frame()

    assert tracker.initialize(frame, (100, 60, 100, 100))

    blank = np.zeros_like(frame)

    result = tracker.update(blank)

    assert result is None
    assert tracker.is_tracking is False
