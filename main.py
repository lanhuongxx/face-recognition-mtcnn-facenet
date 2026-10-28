
import cv2

from src.camera import Camera
from src.face_recognizer import FaceRecognizer
from src.config import REFERENCE_PATH, SIMILARITY_THRESHOLD
from src.visualizer import draw_results


def main():
    # Step 1: Initialize recognizer
    recognizer = FaceRecognizer(
        threshold=SIMILARITY_THRESHOLD
    )

    # Step 2: Register reference face
    reference = cv2.imread(REFERENCE_PATH)

    if reference is None:
        raise FileNotFoundError(
            f"Cannot read reference: {REFERENCE_PATH}"
        )

    recognizer.register(reference)
    print("Reference registered successfully")

    # Step 3: Initialize camera
    camera = Camera()

    try:
        camera.open()

        while True:
            success, frame = camera.read_frame()

            if not success or frame is None:
                print("Cannot read camera frame")
                break

            # Step 4: Recognize faces
            results = recognizer.recognize(frame)

            # Step 5: Visualize results
            display_frame = draw_results(
                frame.copy(),
                results
            )

            cv2.imshow(
                "Real-time Face Recognition",
                display_frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()