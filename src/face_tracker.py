
import cv2
import numpy as np


class FaceTracker:
    """Track a single face using OpenCV MIL Tracker."""

    def __init__(self):
        self.tracker = None
        self.is_tracking = False

    def initialize(self, frame, box):
        """
        Initialize tracker from a detected face.

        Args:
            frame: Current BGR image.
            box: Bounding box (x, y, width, height).

        Returns:
            bool: Whether initialization succeeded.
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame must not be empty")

        if len(box) != 4:
            raise ValueError("Box must contain 4 values")

        x, y, w, h = map(int, box)

        if w <= 0 or h <= 0:
            raise ValueError("Invalid bounding box size")

        frame_h, frame_w = frame.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box outside frame")

        self.reset()
        self.tracker = cv2.TrackerMIL_create()

        self.tracker.init(
            frame,
            (x1, y1, x2 - x1, y2 - y1)
        )

        self.is_tracking = True
        return True

    def update(self, frame):
        """
        Update tracked face position.

        Returns:
            tuple or None: (x, y, width, height)
        """
        if not self.is_tracking or self.tracker is None:
            return None

        if frame is None or frame.size == 0:
            raise ValueError("Frame must not be empty")

        success, box = self.tracker.update(frame)

        if not success:
            self.reset()
            return None

        x, y, w, h = map(int, box)

        frame_h, frame_w = frame.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)

        if x2 <= x1 or y2 <= y1:
            self.reset()
            return None

        return (x1, y1, x2 - x1, y2 - y1)

    def reset(self):
        """Stop tracking and clear tracker state."""
        self.tracker = None
        self.is_tracking = False


class OpticalFlowTracker:
    """Track a single face using Lucas-Kanade Optical Flow + RANSAC."""

    MIN_POINTS = 8
    MIN_INLIER_RATIO = 0.60
    MIN_SCALE = 0.85
    MAX_SCALE = 1.15

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear tracking state."""
        self.box = None
        self.previous_gray = None
        self.previous_points = None
        self.is_tracking = False

    def _detect_features(self, gray, box):
        """Detect feature points near the center of the face."""
        x, y, w, h = box

        mask = np.zeros_like(gray)

        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)

        x1 = x + margin_x
        y1 = y + margin_y
        x2 = x + w - margin_x
        y2 = y + h - margin_y

        mask[y1:y2, x1:x2] = 255

        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=5,
            mask=mask,
        )

    def initialize(self, frame, box):
        """Initialize tracking from a detected face box."""
        if frame is None or frame.size == 0:
            raise ValueError("Frame must not be empty")

        if len(box) != 4:
            raise ValueError("Box must contain 4 values")

        x, y, w, h = map(int, box)

        if w <= 0 or h <= 0:
            raise ValueError("Invalid bounding box size")

        frame_h, frame_w = frame.shape[:2]

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + w)
        y2 = min(frame_h, y + h)

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Bounding box outside frame")

        self.reset()

        self.box = (x1, y1, x2 - x1, y2 - y1)
        self.previous_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY,
        )

        self.previous_points = self._detect_features(
            self.previous_gray,
            self.box,
        )

        if (
            self.previous_points is None
            or len(self.previous_points) < self.MIN_POINTS
        ):
            self.reset()
            return False

        self.is_tracking = True
        return True

    def update(self, frame):
        """Update tracked face position and size."""
        if not self.is_tracking:
            return None

        if frame is None or frame.size == 0:
            raise ValueError("Frame must not be empty")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            self.previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        if new_points is None or status is None:
            self.reset()
            return None

        valid = status.flatten() == 1

        old = self.previous_points[valid].reshape(-1, 2)
        new = new_points[valid].reshape(-1, 2)

        if len(new) < self.MIN_POINTS:
            self.reset()
            return None

        matrix, inliers = cv2.estimateAffinePartial2D(
            old,
            new,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

        if matrix is None or inliers is None:
            self.reset()
            return None

        inlier_mask = inliers.flatten() == 1
        inlier_ratio = np.mean(inlier_mask)

        if inlier_ratio < self.MIN_INLIER_RATIO:
            self.reset()
            return None

        tracked_points = new[inlier_mask].reshape(-1, 1, 2)

        a = matrix[0, 0]
        b = matrix[1, 0]
        scale = np.sqrt(a * a + b * b)

        if not self.MIN_SCALE <= scale <= self.MAX_SCALE:
            self.reset()
            return None

        x, y, w, h = self.box
        frame_h, frame_w = gray.shape

        # Update center using the affine transformation.
        center = np.array(
            [x + w / 2, y + h / 2, 1.0],
            dtype=np.float64,
        )

        new_center = matrix @ center

        # Update size without accumulating rotation-based expansion.
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        new_w = max(30, min(new_w, frame_w))
        new_h = max(30, min(new_h, frame_h))

        x1 = int(round(new_center[0] - new_w / 2))
        y1 = int(round(new_center[1] - new_h / 2))

        x1 = max(0, min(x1, frame_w - new_w))
        y1 = max(0, min(y1, frame_h - new_h))

        self.box = (x1, y1, new_w, new_h)
        self.previous_points = tracked_points
        self.previous_gray = gray.copy()

        return self.box
