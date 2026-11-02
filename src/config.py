from pathlib import Path

# Reference image used to register the known person
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = str(
    PROJECT_ROOT / "data" / "reference" / "my_face.jpg"
)

# Face recognition threshold (cosine similarity)
SIMILARITY_THRESHOLD = 0.70

# Camera device index
CAMERA_ID = 0

# Minimum accepted face size in pixels
MIN_FACE_SIZE = 50

# Minimum MTCNN detection confidence
MIN_CONFIDENCE = 0.90