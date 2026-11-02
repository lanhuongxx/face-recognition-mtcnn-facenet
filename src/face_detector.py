# Module 2 - Face Detection bằng MTCNN
import cv2
import numpy as np
from mtcnn import MTCNN
from src.mtcnn_compat import apply_mtcnn_patch

apply_mtcnn_patch()


class FaceDetector:
    """Phát hiện khuôn mặt bằng MTCNN."""

    def __init__(self):
        self.detector = MTCNN()

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Nhận ảnh BGR và trả về danh sách khuôn mặt."""

        # Step 1: Validate input
        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty")

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Input frame must be a 3-channel BGR image"
            )

        # Step 2: Convert BGR to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Step 3: Detect faces
        faces = self.detector.detect_faces(rgb_frame)

        return faces