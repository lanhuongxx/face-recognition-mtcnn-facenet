
import cv2
import numpy as np

from deepface import DeepFace

from src.face_detector import FaceDetector
from src.face_alignment import align_face
from src.face_matcher import FaceMatcher


def crop_face(image, face):
    x, y, w, h = face["box"]
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
    result = DeepFace.represent(
        img_path=face_image,
        model_name="Facenet",
        detector_backend="skip",
        enforce_detection=False,
        align=False,
        normalization=normalization
    )

    return np.asarray(
        result[0]["embedding"],
        dtype=np.float32
    )


def main():
    image_a = cv2.imread("data/reference/person_01.jpg")
    image_b = cv2.imread("data/reference/person_02.jpg")

    if image_a is None or image_b is None:
        raise FileNotFoundError("Cannot read test images")

    detector = FaceDetector()
    matcher = FaceMatcher(threshold=0.7)

    face_a = detector.detect(image_a)[0]
    face_b = detector.detect(image_b)[0]

    crops = {
        "No alignment": (
            crop_face(image_a, face_a),
            crop_face(image_b, face_b)
        ),
        "Alignment": (
            align_face(image_a, face_a),
            align_face(image_b, face_b)
        )
    }

    print("\n=== ALIGNMENT EXPERIMENT ===")

    for method, (crop_a, crop_b) in crops.items():
        for normalization in ["base", "Facenet"]:
            embedding_a = get_embedding(crop_a, normalization)
            embedding_b = get_embedding(crop_b, normalization)

            result = matcher.match(embedding_a, embedding_b)

            print(
                f"{method:15} | "
                f"{normalization:7} | "
                f"Similarity: {result['similarity']:.4f} | "
                f"{result['label']}"
            )


if __name__ == "__main__":
    main()
