
import numpy as np

from src.face_detector import FaceDetector
from src.face_alignment import align_face
from src.face_embedder import FaceEmbedder
from src.face_matcher import FaceMatcher


class FaceRecognizer:
    def __init__(self, threshold: float = 0.7):
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.matcher = FaceMatcher(threshold=threshold)

        self.reference_embedding = None

    def register(self, image: np.ndarray) -> None:
        if image is None or image.size == 0:
            raise ValueError("Reference image is empty")

        faces = self.detector.detect(image)
        
        # Loại bỏ các vùng quá nhỏ trong ảnh tham chiếu
        min_face_size = 50
        
        valid_faces = [
             face for face in faces
             if face["box"][2] >= min_face_size
             and face["box"][3] >= min_face_size
             ]
        
        print(f"Detected faces: {len(faces)}")
        print(f"Valid faces: {len(valid_faces)}")
        
        if len(valid_faces) != 1:
             raise ValueError(
                  f"Expected exactly one valid reference face, "
                  f"found {len(valid_faces)}"
                  )
        
        aligned_face = align_face(image, valid_faces[0])
        self.reference_embedding = self.embedder.extract(aligned_face)

    def recognize(self, frame: np.ndarray) -> list[dict]:
        if self.reference_embedding is None:
            raise RuntimeError("Reference face is not registered")

        faces = self.detector.detect(frame)
        results = []

        for face in faces:
            aligned_face = align_face(frame, face)
            embedding = self.embedder.extract(aligned_face)

            match_result = self.matcher.match(
                self.reference_embedding,
                embedding
            )

            results.append({
                "box": face["box"],
                "similarity": match_result["similarity"],
                "label": match_result["label"]
            })

        return results
