# Module 3 - FaceNet: Trích xuất đặc trưng khuôn mặt (Face Embedding)
import numpy as np
from deepface import DeepFace


class FaceEmbedder:
    """Trích xuất vector đặc trưng khuôn mặt bằng FaceNet."""

    def __init__(self):
        self.model_name = "Facenet"

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """Trả về embedding FaceNet có kích thước (128,)."""

        # Step 1: Validate input
        if face_image is None or face_image.size == 0:
            raise ValueError("Face image is empty")

        if face_image.ndim != 3 or face_image.shape[2] != 3:
            raise ValueError(
                "Face image must have 3 color channels"
            )

        # Step 2: Extract embedding
        results = DeepFace.represent(
            img_path=face_image,
            model_name=self.model_name,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
            normalization="Facenet",
        )

        # Step 3: Validate DeepFace output
        if not results or "embedding" not in results[0]:
            raise ValueError(
                "DeepFace did not return a valid embedding"
            )

        embedding = np.asarray(
            results[0]["embedding"],
            dtype=np.float32,
        )

        if embedding.shape != (128,):
            raise ValueError(
                f"Expected embedding shape (128,), "
                f"got {embedding.shape}"
            )

        if not np.isfinite(embedding).all():
            raise ValueError(
                "Embedding contains NaN or infinity"
            )

        return embedding