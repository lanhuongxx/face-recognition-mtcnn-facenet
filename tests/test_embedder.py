
import numpy as np
import pytest
from unittest.mock import patch

from src.face_embedder import FaceEmbedder


@patch("src.face_embedder.DeepFace")
def test_extract_empty_image(mock_deepface):
    embedder = FaceEmbedder()

    with pytest.raises(ValueError):
        embedder.extract(None)


@patch("src.face_embedder.DeepFace")
def test_extract_invalid_channels(mock_deepface):
    embedder = FaceEmbedder()

    gray_image = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(ValueError):
        embedder.extract(gray_image)


@patch("src.face_embedder.DeepFace")
def test_extract_embedding_shape(mock_deepface):
    fake_embedding = [0.1] * 128

    mock_deepface.represent.return_value = [
        {"embedding": fake_embedding}
    ]

    embedder = FaceEmbedder()

    face_image = np.zeros(
        (160, 160, 3),
        dtype=np.uint8
    )

    embedding = embedder.extract(face_image)

    assert embedding.shape == (128,)
    assert embedding.dtype == np.float32
    assert np.isfinite(embedding).all()
    mock_deepface.represent.assert_called_once()
