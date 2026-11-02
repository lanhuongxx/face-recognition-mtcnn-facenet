import numpy as np
import cv2
import time

from src.config import (
    SIMILARITY_THRESHOLD,
    MIN_FACE_SIZE,
    MIN_CONFIDENCE,
)
from src.face_detector import FaceDetector
from src.face_alignment import align_face
from src.face_embedder import FaceEmbedder
from src.face_matcher import FaceMatcher


class FaceRecognizer:
    """Điều phối pipeline đăng ký và nhận diện khuôn mặt."""

    def __init__(
        self,
        threshold: float = SIMILARITY_THRESHOLD,
        min_face_size: int = MIN_FACE_SIZE,
        min_confidence: float = MIN_CONFIDENCE,
    ):
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.matcher = FaceMatcher(threshold=threshold)

        self.min_face_size = min_face_size
        self.min_confidence = min_confidence

        self.reference_embedding = None

    def _is_valid_face(self, face: dict) -> bool:
        """Kiểm tra kích thước và độ tin cậy của khuôn mặt."""

        box = face["box"]
        confidence = face["confidence"]

        width = box[2]
        height = box[3]

        return (
            width >= self.min_face_size
            and height >= self.min_face_size
            and confidence >= self.min_confidence
        )

    def register(self, image: np.ndarray) -> None:
        """Đăng ký embedding từ ảnh chứa đúng một khuôn mặt hợp lệ."""

        if image is None or image.size == 0:
            raise ValueError("Reference image is empty")

        faces = self.detector.detect(image)

        valid_faces = [
            face for face in faces
            if self._is_valid_face(face)
        ]

        print(f"Detected faces: {len(faces)}")
        print(f"Valid faces: {len(valid_faces)}")

        if len(valid_faces) != 1:
            raise ValueError(
                "Expected exactly one valid reference face, "
                f"found {len(valid_faces)}"
            )

        aligned_face = align_face(image, valid_faces[0])
        embedding = self.embedder.extract(aligned_face)

        self.reference_embedding = embedding

    
    def recognize(self, frame: np.ndarray) -> list[dict]:
        if self.reference_embedding is None:
            raise RuntimeError("Reference face is not registered")

        if frame is None or frame.size == 0:
            return []

        # Step 1: Measure MTCNN detection
        start = time.perf_counter()
        faces = self.detector.detect(frame)
        detection_time = time.perf_counter() - start

        alignment_time = 0.0
        embedding_time = 0.0
        matching_time = 0.0

        results = []

        for face in faces:
            if not self._is_valid_face(face):
                continue

            try:
                # Step 2: Measure face alignment
                start = time.perf_counter()
                aligned_face = align_face(frame, face)
                alignment_time += time.perf_counter() - start

                # Step 3: Measure FaceNet embedding
                start = time.perf_counter()
                embedding = self.embedder.extract(aligned_face)
                embedding_time += time.perf_counter() - start

                # Step 4: Measure cosine similarity
                start = time.perf_counter()
                match_result = self.matcher.match(
                    self.reference_embedding,
                    embedding
                )
                matching_time += time.perf_counter() - start

            except (ValueError, cv2.error) as error:
                print(f"Skipping invalid face: {error}")
                continue

            results.append({
                "box": face["box"],
                "similarity": match_result["similarity"],
                "label": match_result["label"]
            })

        print(
            f"MTCNN: {detection_time:.3f}s | "
            f"Alignment: {alignment_time:.3f}s | "
            f"FaceNet: {embedding_time:.3f}s | "
            f"Matching: {matching_time:.4f}s"
        )

        return results
