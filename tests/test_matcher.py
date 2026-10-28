
import numpy as np
import pytest

from src.face_matcher import FaceMatcher


def test_identical_vectors():
    matcher = FaceMatcher()
    a = np.array([1.0, 2.0, 3.0])

    result = matcher.match(a, a)

    assert result["similarity"] == pytest.approx(1.0)
    assert result["label"] == "Matched"


def test_orthogonal_vectors():
    matcher = FaceMatcher()

    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])

    result = matcher.match(a, b)

    assert result["similarity"] == pytest.approx(0.0)
    assert result["label"] == "Unknown"


def test_threshold_boundary():
    matcher = FaceMatcher(threshold=0.7)

    a = np.array([1.0, 0.0])
    b = np.array([0.7, np.sqrt(0.51)])

    result = matcher.match(a, b)

    assert result["similarity"] == pytest.approx(0.7)
    assert result["label"] == "Matched"


def test_zero_vector():
    matcher = FaceMatcher()

    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])

    with pytest.raises(ValueError):
        matcher.cosine_similarity(a, b)


def test_different_shapes():
    matcher = FaceMatcher()

    a = np.array([1.0, 2.0])
    b = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        matcher.cosine_similarity(a, b)
