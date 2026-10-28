
import cv2

from src.face_detector import FaceDetector
from src.face_alignment import align_face


detector = FaceDetector()

for filename in ["person_01.jpg", "person_02.jpg"]:
    image = cv2.imread(f"data/reference/{filename}")

    if image is None:
        raise FileNotFoundError(filename)

    faces = detector.detect(image)

    if not faces:
        raise RuntimeError(f"No face found in {filename}")

    aligned = align_face(image, faces[0])

    print(filename, "aligned shape:", aligned.shape)

    cv2.imshow(f"Aligned - {filename}", aligned)

cv2.waitKey(0)
cv2.destroyAllWindows()
