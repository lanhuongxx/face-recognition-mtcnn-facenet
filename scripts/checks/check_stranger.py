
import cv2
import numpy as np

from deepface import DeepFace

from src.face_detector import FaceDetector
from src.face_alignment import align_face
from src.face_matcher import FaceMatcher


def extract_embedding(path, detector):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(path)

    faces = detector.detect(image)

    if len(faces) != 1:
        raise ValueError(
            f"Expected 1 face in {path}, found {len(faces)}"
        )

    aligned = align_face(image, faces[0])

    result = DeepFace.represent(
        img_path=aligned,
        model_name="Facenet",
        detector_backend="skip",
        enforce_detection=False,
        align=False,
        normalization="Facenet"
    )

    return np.asarray(
        result[0]["embedding"],
        dtype=np.float32
    )


def main():
    detector = FaceDetector()
    matcher = FaceMatcher(threshold=0.7)

    reference = extract_embedding(
        "data/reference/person_01.jpg",
        detector
    )

    same_person = extract_embedding(
        "data/reference/person_02.jpg",
        detector
    )

    stranger = extract_embedding(
        "data/reference/stranger_01.jpg",
        detector
    )

    for name, embedding in [
        ("Same person", same_person),
        ("Different person", stranger)
    ]:
        result = matcher.match(reference, embedding)

        print(
            f"{name:17} | "
            f"Similarity: {result['similarity']:.4f} | "
            f"{result['label']}"
        )


if __name__ == "__main__":
    main()