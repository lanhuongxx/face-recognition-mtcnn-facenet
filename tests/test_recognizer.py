
import numpy as np
import pytest

from unittest.mock import MagicMock, patch

from src.face_recognizer import FaceRecognizer


@pytest.fixture
def recognizer():
    """Tạo FaceRecognizer với các dependency được mock."""

    with (
        patch("src.face_recognizer.FaceDetector"),
        patch("src.face_recognizer.FaceEmbedder"),
        patch("src.face_recognizer.FaceMatcher"),
    ):
        recognizer = FaceRecognizer(
            threshold=0.7,
            min_face_size=50,
            min_confidence=0.90,
        )

        yield recognizer


@pytest.fixture
def sample_image():
    """Ảnh giả lập dùng trong unit test."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def sample_embedding():
    """Embedding FaceNet giả lập gồm 128 chiều."""
    return np.ones(128, dtype=np.float32)


def make_face(
    box=None,
    confidence=0.99,
):
    """Tạo kết quả phát hiện khuôn mặt giả lập."""

    if box is None:
        box = [10, 10, 50, 50]

    return {
        "box": box,
        "confidence": confidence,
    }


# ============================================================
# 1. Registration tests
# ============================================================

def test_register_empty_image(recognizer):
    """Không cho phép đăng ký ảnh rỗng."""

    with pytest.raises(ValueError, match="Reference image is empty"):
        recognizer.register(None)

    with pytest.raises(ValueError, match="Reference image is empty"):
        recognizer.register(np.array([]))


def test_register_no_face(recognizer, sample_image):
    """Không đăng ký khi MTCNN không phát hiện khuôn mặt."""

    recognizer.detector.detect.return_value = []

    with pytest.raises(
        ValueError,
        match="Expected exactly one valid reference face",
    ):
        recognizer.register(sample_image)


def test_register_one_face(
    recognizer,
    sample_image,
    sample_embedding,
):
    """Đăng ký thành công khi có đúng một khuôn mặt hợp lệ."""

    recognizer.detector.detect.return_value = [
        make_face(
            box=[10, 10, 50, 50],
            confidence=0.99,
        )
    ]

    recognizer.embedder.extract.return_value = sample_embedding

    with patch(
        "src.face_recognizer.align_face",
        return_value=sample_image,
    ):
        recognizer.register(sample_image)

    np.testing.assert_array_equal(
        recognizer.reference_embedding,
        sample_embedding,
    )

    recognizer.embedder.extract.assert_called_once()


def test_register_multiple_faces(recognizer, sample_image):
    """Từ chối ảnh chứa hai khuôn mặt hợp lệ."""

    recognizer.detector.detect.return_value = [
        make_face(
            box=[0, 0, 50, 50],
            confidence=0.99,
        ),
        make_face(
            box=[50, 0, 50, 50],
            confidence=0.98,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="found 2",
    ):
        recognizer.register(sample_image)

    recognizer.embedder.extract.assert_not_called()


def test_register_low_confidence(recognizer, sample_image):
    """Từ chối khuôn mặt có confidence thấp."""

    recognizer.detector.detect.return_value = [
        make_face(
            box=[10, 10, 80, 80],
            confidence=0.50,
        )
    ]

    with pytest.raises(ValueError, match="found 0"):
        recognizer.register(sample_image)


def test_register_small_face(recognizer, sample_image):
    """Từ chối khuôn mặt nhỏ hơn kích thước tối thiểu."""

    recognizer.detector.detect.return_value = [
        make_face(
            box=[10, 10, 30, 30],
            confidence=0.99,
        )
    ]

    with pytest.raises(ValueError, match="found 0"):
        recognizer.register(sample_image)


def test_register_failure_keeps_old_embedding(
    recognizer,
    sample_image,
    sample_embedding,
):
    """Đăng ký thất bại không làm mất embedding cũ."""

    old_embedding = np.full(128, 2.0, dtype=np.float32)
    recognizer.reference_embedding = old_embedding.copy()

    recognizer.detector.detect.return_value = [
        make_face()
    ]

    recognizer.embedder.extract.side_effect = ValueError(
        "Invalid face embedding"
    )

    with patch(
        "src.face_recognizer.align_face",
        return_value=sample_image,
    ):
        with pytest.raises(ValueError):
            recognizer.register(sample_image)

    np.testing.assert_array_equal(
        recognizer.reference_embedding,
        old_embedding,
    )


# ============================================================
# 2. Recognition tests
# ============================================================

def test_recognize_without_registration(
    recognizer,
    sample_image,
):
    """Không nhận diện nếu chưa đăng ký khuôn mặt."""

    with pytest.raises(
        RuntimeError,
        match="Reference face is not registered",
    ):
        recognizer.recognize(sample_image)


def test_recognize_empty_frame(
    recognizer,
    sample_embedding,
):
    """Frame rỗng trả về danh sách kết quả rỗng."""

    recognizer.reference_embedding = sample_embedding

    assert recognizer.recognize(None) == []
    assert recognizer.recognize(np.array([])) == []

    recognizer.detector.detect.assert_not_called()


def test_recognize_no_face(
    recognizer,
    sample_image,
    sample_embedding,
):
    """Không có khuôn mặt thì không có kết quả nhận diện."""

    recognizer.reference_embedding = sample_embedding
    recognizer.detector.detect.return_value = []

    results = recognizer.recognize(sample_image)

    assert results == []


def test_recognize_matched_face(
    recognizer,
    sample_image,
    sample_embedding,
):
    """Khuôn mặt có similarity cao được gán nhãn Matched."""

    recognizer.reference_embedding = sample_embedding

    recognizer.detector.detect.return_value = [
        make_face(
            box=[10, 20, 50, 60],
            confidence=0.99,
        )
    ]

    recognizer.embedder.extract.return_value = sample_embedding

    recognizer.matcher.match.return_value = {
        "similarity": 0.85,
        "label": "Matched",
    }

    with patch(
        "src.face_recognizer.align_face",
        return_value=sample_image,
    ):
        results = recognizer.recognize(sample_image)

    assert len(results) == 1
    assert results[0]["box"] == [10, 20, 50, 60]
    assert results[0]["similarity"] == 0.85
    assert results[0]["label"] == "Matched"


def test_recognize_unknown_face(
    recognizer,
    sample_image,
    sample_embedding,
):
    """Khuôn mặt không khớp được gán nhãn Unknown."""

    recognizer.reference_embedding = sample_embedding

    recognizer.detector.detect.return_value = [
        make_face()
    ]

    recognizer.embedder.extract.return_value = sample_embedding

    recognizer.matcher.match.return_value = {
        "similarity": 0.35,
        "label": "Unknown",
    }

    with patch(
        "src.face_recognizer.align_face",
        return_value=sample_image,
    ):
        results = recognizer.recognize(sample_image)

    assert len(results) == 1
    assert results[0]["label"] == "Unknown"
    assert results[0]["similarity"] == 0.35


def test_recognize_filters_invalid_faces(
    recognizer,
    sample_image,
    sample_embedding,
):
    """Chỉ xử lý khuôn mặt đạt ngưỡng kích thước và confidence."""

    recognizer.reference_embedding = sample_embedding

    recognizer.detector.detect.return_value = [
        make_face(box=[0, 0, 80, 80], confidence=0.99),
        make_face(box=[0, 0, 80, 80], confidence=0.50),
        make_face(box=[0, 0, 20, 20], confidence=0.99),
    ]

    recognizer.embedder.extract.return_value = sample_embedding

    recognizer.matcher.match.return_value = {
        "similarity": 0.90,
        "label": "Matched",
    }

    with patch(
        "src.face_recognizer.align_face",
        return_value=sample_image,
    ) as mock_align:
        results = recognizer.recognize(sample_image)

    assert len(results) == 1
    assert results[0]["label"] == "Matched"

    mock_align.assert_called_once()
    recognizer.embedder.extract.assert_called_once()


def test_recognize_skips_alignment_error(
    recognizer,
    sample_image,
    sample_embedding,
):
    """Lỗi căn chỉnh một khuôn mặt không dừng toàn bộ quá trình."""

    recognizer.reference_embedding = sample_embedding

    recognizer.detector.detect.return_value = [
        make_face(box=[0, 0, 50, 50]),
        make_face(box=[50, 0, 50, 50]),
    ]

    recognizer.embedder.extract.return_value = sample_embedding

    recognizer.matcher.match.return_value = {
        "similarity": 0.90,
        "label": "Matched",
    }

    with patch(
        "src.face_recognizer.align_face",
        side_effect=[
            ValueError("Invalid crop"),
            sample_image,
        ],
    ):
        results = recognizer.recognize(sample_image)

    assert len(results) == 1
    assert results[0]["box"] == [50, 0, 50, 50]
    assert results[0]["label"] == "Matched"


# ============================================================
# 3. Face validation tests
# ============================================================

def test_valid_face(recognizer):
    """Khuôn mặt đủ kích thước và confidence được chấp nhận."""

    face = make_face(
        box=[10, 10, 100, 100],
        confidence=0.99,
    )

    assert recognizer._is_valid_face(face) is True


def test_invalid_face_low_confidence(recognizer):
    """Khuôn mặt confidence thấp bị loại."""

    face = make_face(
        box=[10, 10, 100, 100],
        confidence=0.50,
    )

    assert recognizer._is_valid_face(face) is False


def test_invalid_face_small_size(recognizer):
    """Khuôn mặt quá nhỏ bị loại."""

    face = make_face(
        box=[10, 10, 30, 30],
        confidence=0.99,
    )

    assert recognizer._is_valid_face(face) is False


def test_valid_face_at_threshold(recognizer):
    """Khuôn mặt đúng bằng các ngưỡng vẫn được chấp nhận."""

    face = make_face(
        box=[10, 10, 50, 50],
        confidence=0.90,
    )

    assert recognizer._is_valid_face(face) is True
