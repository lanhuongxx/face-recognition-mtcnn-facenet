# Module 3 - FaceNet: Trích xuất đặc trưng khuôn mặt (Face Embedding)
import numpy as np
from deepface import DeepFace


class FaceEmbedder:
    def __init__(self):
        self.model_name = "Facenet"
        self.model = DeepFace.build_model(self.model_name)

    def extract(self, face_image: np.ndarray) -> np.ndarray:
        if face_image is None or face_image.size == 0:
            raise ValueError("Face image is empty")

        if face_image.ndim != 3 or face_image.shape[2] != 3:
            raise ValueError("Face image must have 3 color channels")

        results = DeepFace.represent(
            img_path=face_image,
            model_name=self.model_name,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
            normalization="Facenet"
        )

        embedding = np.asarray(
            results[0]["embedding"],
            dtype=np.float32
        )

        return embedding
