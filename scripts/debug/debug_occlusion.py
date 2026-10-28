
import cv2
import numpy as np

from src.face_detector import FaceDetector
from src.face_alignment import align_face
from src.face_embedder import FaceEmbedder
from src.face_matcher import FaceMatcher


REFERENCE_PATH = "data/reference/my_face.jpg"

TEST_IMAGES = {
    "Normal": "data/evaluation/normal.jpg",
    "Eye occlusion": "data/evaluation/eye_occlusion.jpg",
}


def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    return image


def extract_face(image, detector, use_alignment):
    faces = detector.detect(image)

    # Bỏ các vùng quá nhỏ, tương tự bước đăng ký tham chiếu
    valid_faces = [
        face for face in faces
        if face["box"][2] >= 50 and face["box"][3] >= 50
    ]
    
    print(f"\nDetected faces: {len(faces)}")
    print(f"Valid faces: {len(valid_faces)}")
    
    for i, face in enumerate(faces):
        print(
            f"Face {i}: "
            f"box={face['box']}, "
            f"confidence={face['confidence']:.4f}"
            )
        
        if len(valid_faces) != 1:
            raise ValueError(
                f"Expected 1 valid face, found {len(valid_faces)}"
                )

    face = valid_faces[0]

    if use_alignment:
        return align_face(image, face)

    # Không alignment: chỉ crop bounding box
    x, y, w, h = face["box"]
    height, width = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    cropped = image[y1:y2, x1:x2]

    if cropped.size == 0:
        raise ValueError("Empty face crop")

    return cropped


def main():
    detector = FaceDetector()
    embedder = FaceEmbedder()
    matcher = FaceMatcher(threshold=0.7)

    reference_image = load_image(REFERENCE_PATH)

    for use_alignment in [True, False]:
        mode = "WITH ALIGNMENT" if use_alignment else "NO ALIGNMENT"

        print(f"\n===== {mode} =====")

        reference_face = extract_face(
            reference_image, detector, use_alignment
        )
        reference_embedding = embedder.extract(reference_face)

        for name, path in TEST_IMAGES.items():
            print(f"\nProcessing: {name} ({path})")
            image = load_image(path)

            test_face = extract_face(
                image, detector, use_alignment
            )
            test_embedding = embedder.extract(test_face)

            result = matcher.match(
                reference_embedding, test_embedding
            )

            print(
                f"{name:20s} | "
                f"Similarity: {result['similarity']:.4f} | "
                f"{result['label']}"
            )


if __name__ == "__main__":
    main()
