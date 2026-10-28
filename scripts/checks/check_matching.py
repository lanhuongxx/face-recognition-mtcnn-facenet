# Module 5 - Tích hợp nhận diện khuôn mặt realtime 
import cv2

from src.face_detector import FaceDetector
from src.face_embedder import FaceEmbedder
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

    face_crop = image[y1:y2, x1:x2]

    if face_crop.size == 0:
        raise ValueError("Face crop is empty")

    return face_crop


def main():
    image_a = cv2.imread("data/reference/person_01.jpg")
    image_b = cv2.imread("data/reference/person_02.jpg")

    if image_a is None or image_b is None:
        raise FileNotFoundError("Cannot read reference images")

    detector = FaceDetector()
    embedder = FaceEmbedder()
    matcher = FaceMatcher(threshold=0.7)

    face_a = crop_first_face(image_a, detector)
    face_b = crop_first_face(image_b, detector)

    print("Cropped face A shape:", face_a.shape)
    print("Cropped face B shape:", face_b.shape)

    cv2.imshow("Cropped Face A", face_a)
    cv2.imshow("Cropped Face B", face_b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    embedding_a = embedder.extract(face_a)
    embedding_b = embedder.extract(face_b)

    result = matcher.match(embedding_a, embedding_b)

    print("Embedding A:", embedding_a.shape)
    print("Embedding B:", embedding_b.shape)
    print("Similarity:", round(result["similarity"], 4))
    print("Result:", result["label"])


if __name__ == "__main__":
    main()
