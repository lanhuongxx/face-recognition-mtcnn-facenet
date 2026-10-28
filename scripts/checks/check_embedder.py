
import cv2
import numpy as np

from src.face_detector import FaceDetector
from src.face_embedder import FaceEmbedder


image = cv2.imread("data/reference/person_01.jpg")

if image is None:
    raise FileNotFoundError("Reference image not found")

detector = FaceDetector()
faces = detector.detect(image)

if not faces:
    raise RuntimeError("No face detected")

x, y, w, h = faces[0]["box"]

height, width = image.shape[:2]

x1 = max(0, x)
y1 = max(0, y)
x2 = min(width, x + w)
y2 = min(height, y + h)

face_crop = image[y1:y2, x1:x2]

if face_crop.size == 0:
    raise ValueError("Face crop is empty")

embedder = FaceEmbedder()
embedding = embedder.extract(face_crop)

print("Embedding shape:", embedding.shape)
print("Embedding dtype:", embedding.dtype)
print("First 10 values:", embedding[:10])
print("All finite:", np.isfinite(embedding).all())
