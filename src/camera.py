# Module 1 - Camera Management
import cv2


class Camera:
    """Quản lý việc mở, đọc frame và giải phóng webcam."""

    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None

    def open(self) -> bool:
        """Mở webcam và trả về True nếu thành công."""

        # Nếu camera đã mở thì không tạo kết nối mới
        if self.cap is not None and self.cap.isOpened():
            return True

        # Giải phóng đối tượng cũ nếu có
        self.release()

        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            self.release()
            return False

        return True

    def read_frame(self):
        """Đọc một frame từ webcam.

        Returns:
            tuple: (success, frame)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        success, frame = self.cap.read()

        if not success or frame is None:
            return False, None

        return True, frame

    def release(self):
        """Giải phóng tài nguyên webcam."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
