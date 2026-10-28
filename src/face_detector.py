# Module 2 - Face Detection bằng MTCNN
import cv2
import numpy as np
from mtcnn import MTCNN


class FaceDetector:
    """Phát hiện khuôn mặt bằng MTCNN."""

    def __init__(self):
        self.detector = MTCNN()

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Nhận ảnh BGR và trả về danh sách khuôn mặt."""

        if frame is None or frame.size == 0:
            raise ValueError("Input frame is empty")

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        faces = self.detector.detect_faces(rgb_frame)

        return faces
