
from unittest.mock import MagicMock, patch

from src.camera import Camera


def test_read_without_open():
    camera = Camera()

    success, frame = camera.read_frame()

    assert success is False
    assert frame is None


def test_open_camera():
    with patch("src.camera.cv2.VideoCapture") as mock_capture:
        mock_capture.return_value.isOpened.return_value = True

        camera = Camera()
        result = camera.open()

        assert result is True


def test_release_camera():
    camera = Camera()
    mock_cap = MagicMock()
    camera.cap = mock_cap

    camera.release()

    mock_cap.release.assert_called_once()
    assert camera.cap is None
