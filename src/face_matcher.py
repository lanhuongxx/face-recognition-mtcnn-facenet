# Module 4 - Face Matching bằng Cosine Similarity
import numpy as np


class FaceMatcher:
    def __init__(self, threshold: float = 0.7):
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between -1 and 1")

        self.threshold = threshold

    def cosine_similarity(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray
    ) -> float:

        a = np.asarray(embedding_a, dtype=np.float64)
        b = np.asarray(embedding_b, dtype=np.float64)

        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("Embeddings must be 1D vectors")

        if a.shape != b.shape or a.size == 0:
            raise ValueError("Embeddings must have the same nonzero shape")

        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Embeddings must contain finite values")

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            raise ValueError("Embedding norm cannot be zero")

        similarity = np.dot(a, b) / (norm_a * norm_b)

        return float(np.clip(similarity, -1.0, 1.0))

    def match(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray
    ) -> dict:

        similarity = self.cosine_similarity(
            embedding_a,
            embedding_b
        )

        return {
            "similarity": similarity,
            "label": "Matched" if similarity >= self.threshold else "Unknown"
        }
