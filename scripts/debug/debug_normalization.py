
import cv2
import numpy as np

from deepface import DeepFace

from src.face_detector import FaceDetector
from src.face_matcher import FaceMatcher


def crop_first_face(image, detector):
    faces = detector.detect(image)

    if not faces:
        raise ValueError("No face detected")

    x, y, w, h = faces[0]["box"]

    height, width = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        raise ValueError("Empty face crop")

    return crop


def get_embedding(face_image, normalization):
    results = DeepFace.represent(
        img_path=face_image,
        model_name="Facenet",
        detector_backend="skip",
        enforce_detection=False,
        align=False,
        normalization=normalization
    )

    return np.asarray(
        results[0]["embedding"],
        dtype=np.float32
    )


def main():
    image_a = cv2.imread("data/reference/person_01.jpg")
    image_b = cv2.imread("data/reference/person_02.jpg")

    if image_a is None or image_b is None:
        raise FileNotFoundError("Cannot read images")

    detector = FaceDetector()
    matcher = FaceMatcher(threshold=0.7)

    face_a = crop_first_face(image_a, detector)
    face_b = crop_first_face(image_b, detector)

    for normalization in ["base", "Facenet"]:
        embedding_a = get_embedding(face_a, normalization)
        embedding_b = get_embedding(face_b, normalization)

        result = matcher.match(embedding_a, embedding_b)

        print(f"\nNormalization: {normalization}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Result: {result['label']}")


if __name__ == "__main__":
    main()
