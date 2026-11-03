
from collections import deque

from src.face_tracker import OpticalFlowTracker


class FrameHistory:
    """Store recent frames for asynchronous face recognition."""

    def __init__(self, max_frames=90):
        if max_frames < 2:
            raise ValueError("max_frames must be at least 2")

        self.frames = deque(maxlen=max_frames)

    def append(self, frame_id, frame):
        """Store a copy of a frame."""
        self.frames.append((frame_id, frame.copy()))

    def get_from(self, start_frame_id):
        """
        Return frames starting from start_frame_id.

        Returns None if the source frame is unavailable.
        """
        items = list(self.frames)

        for index, (frame_id, _) in enumerate(items):
            if frame_id == start_frame_id:
                return items[index:]

        return None

    def clear(self):
        """Remove all stored frames."""
        self.frames.clear()


def propagate_box(history, start_frame_id, detected_box):
    """
    Propagate a detected bounding box through stored frames.

    Uses Lucas-Kanade Optical Flow + RANSAC.

    Args:
        history: FrameHistory instance.
        start_frame_id: ID of the recognition source frame.
        detected_box: (x, y, width, height).

    Returns:
        (latest_frame_id, latest_box) on success.
        None if tracking fails or the source frame is missing.
    """
    frames = history.get_from(start_frame_id)

    if not frames:
        return None

    _, source_frame = frames[0]

    temporary_tracker = OpticalFlowTracker()

    initialized = temporary_tracker.initialize(
        source_frame,
        detected_box,
    )

    if not initialized:
        return None

    for _, frame in frames[1:]:
        updated_box = temporary_tracker.update(frame)

        if updated_box is None:
            return None

    latest_frame_id = frames[-1][0]

    return latest_frame_id, temporary_tracker.box


def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union (IoU).

    Bounding box format:
        (x, y, width, height)

    Returns:
        Float between 0 and 1 for valid boxes.
    """
    x1, y1, w1, h1 = box_a
    x2, y2, w2, h2 = box_b

    left = max(x1, x2)
    top = max(y1, y2)

    right = min(x1 + w1, x2 + w2)
    bottom = min(y1 + h1, y2 + h2)

    intersection = (
        max(0, right - left)
        * max(0, bottom - top)
    )

    area_a = w1 * h1
    area_b = w2 * h2

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def replay_snapshot(frames, start_frame_id, detected_box):
    """
    Replay Optical Flow using a fixed snapshot of frames.

    This allows replay to run in a background worker
    without modifying the live frame history.

    Args:
        frames: List of (frame_id, frame).
        start_frame_id: Original recognition frame ID.
        detected_box: Bounding box from recognition.

    Returns:
        (latest_frame_id, latest_box) or None.
    """
    history = FrameHistory(
        max_frames=max(2, len(frames))
    )

    for frame_id, frame in frames:
        history.append(frame_id, frame)

    return propagate_box(
        history,
        start_frame_id,
        detected_box,
    )

def is_recognition_valid(
    job,
    current_generation,
    current_time,
    max_age=3.0,
):
    """Check whether an asynchronous result is still valid."""
    if job is None:
        return False

    age = current_time - job["submitted_at"]

    return (
        0 <= age <= max_age
        and job["generation"] == current_generation
    )