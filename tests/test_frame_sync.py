
import numpy as np
from unittest.mock import patch

from src.frame_sync import (
    FrameHistory,
    propagate_box,
    calculate_iou,
    replay_snapshot,
    is_recognition_valid,
)


def make_frame():
    """Create a dummy BGR frame."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


# ============================================================
# FrameHistory tests
# ============================================================

def test_history_keeps_recent_frames():
    history = FrameHistory(max_frames=3)

    for frame_id in range(5):
        history.append(frame_id, make_frame())

    assert history.get_from(0) is None

    frames = history.get_from(2)

    assert [frame_id for frame_id, _ in frames] == [
        2, 3, 4
    ]


def test_history_stores_copy():
    history = FrameHistory()
    frame = make_frame()

    history.append(1, frame)

    frame[:] = 255

    saved = history.get_from(1)[0][1]

    assert np.all(saved == 0)


# ============================================================
# propagate_box tests
# ============================================================

def test_propagate_box_success():
    history = FrameHistory()

    for frame_id in range(3):
        history.append(frame_id, make_frame())

    with patch(
        "src.frame_sync.OpticalFlowTracker"
    ) as tracker_cls:

        tracker = tracker_cls.return_value

        tracker.initialize.return_value = True
        tracker.update.return_value = (20, 10, 30, 30)
        tracker.box = (20, 10, 30, 30)

        result = propagate_box(
            history,
            start_frame_id=0,
            detected_box=(10, 10, 30, 30),
        )

    assert result == (
        2,
        (20, 10, 30, 30),
    )

    assert tracker.update.call_count == 2


def test_propagate_box_missing_source():
    history = FrameHistory()

    history.append(5, make_frame())

    result = propagate_box(
        history,
        start_frame_id=0,
        detected_box=(10, 10, 30, 30),
    )

    assert result is None


def test_propagate_box_tracking_failure():
    history = FrameHistory()

    history.append(0, make_frame())
    history.append(1, make_frame())

    with patch(
        "src.frame_sync.OpticalFlowTracker"
    ) as tracker_cls:

        tracker = tracker_cls.return_value

        tracker.initialize.return_value = True
        tracker.update.return_value = None

        result = propagate_box(
            history,
            start_frame_id=0,
            detected_box=(10, 10, 30, 30),
        )

    assert result is None


# ============================================================
# IoU tests
# ============================================================

def test_iou_identical_boxes():
    box = (10, 10, 30, 30)

    assert calculate_iou(box, box) == 1.0


def test_iou_no_overlap():
    result = calculate_iou(
        (0, 0, 10, 10),
        (30, 30, 10, 10),
    )

    assert result == 0.0


def test_iou_partial_overlap():
    result = calculate_iou(
        (0, 0, 20, 20),
        (10, 0, 20, 20),
    )

    expected = 1 / 3

    assert abs(result - expected) < 1e-6


# ============================================================
# replay_snapshot tests
# ============================================================

def test_replay_snapshot_missing_frame():
    frames = [
        (10, make_frame()),
        (11, make_frame()),
    ]

    result = replay_snapshot(
        frames,
        start_frame_id=5,
        detected_box=(10, 10, 30, 30),
    )

    assert result is None


def test_replay_snapshot_tracking_failure():
    frames = [
        (10, make_frame()),
        (11, make_frame()),
    ]

    with patch(
        "src.frame_sync.OpticalFlowTracker"
    ) as tracker_cls:

        tracker = tracker_cls.return_value

        tracker.initialize.return_value = True
        tracker.update.return_value = None

        result = replay_snapshot(
            frames,
            start_frame_id=10,
            detected_box=(10, 10, 30, 30),
        )

    assert result is None

def test_recognition_valid():
    job = {
        "submitted_at": 10.0,
        "generation": 2,
    }

    assert is_recognition_valid(job, 2, 11.0)


def test_recognition_expired():
    job = {
        "submitted_at": 10.0,
        "generation": 2,
    }

    assert not is_recognition_valid(job, 2, 14.0)


def test_recognition_old_generation():
    job = {
        "submitted_at": 10.0,
        "generation": 2,
    }

    assert not is_recognition_valid(job, 3, 11.0)


def test_recognition_missing_job():
    assert not is_recognition_valid(None, 2, 11.0)