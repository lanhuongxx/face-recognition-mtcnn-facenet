# Module 1 - Camera Management
import cv2


class Camera:
    """Quản lý việc truy cập và đọc hình ảnh từ webcam."""

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None

    def open(self) -> bool:
        """Mở webcam và trả về trạng thái kết nối."""
        self.cap = cv2.VideoCapture(self.camera_id)
        return self.cap.isOpened()

    def read_frame(self):
        """Đọc một frame từ webcam."""
        if self.cap is None or not self.cap.isOpened():
            return False, None

        success, frame = self.cap.read()
        return success, frame

    def release(self):
        """Giải phóng tài nguyên webcam."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
