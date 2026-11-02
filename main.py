import cv2
import time

from src.camera import Camera
from src.face_recognizer import FaceRecognizer
from src.config import (
    REFERENCE_PATH,
    SIMILARITY_THRESHOLD,
    CAMERA_ID,
)
from src.visualizer import draw_results


def main():
    # Step 1: Initialize face recognizer
    recognizer = FaceRecognizer(
        threshold=SIMILARITY_THRESHOLD
    )

    # Step 2: Load reference image
    reference = cv2.imread(REFERENCE_PATH)

    if reference is None:
        raise FileNotFoundError(
            f"Cannot read reference: {REFERENCE_PATH}"
        )

    # Step 3: Register reference face
    recognizer.register(reference)
    print("Reference registered successfully")

    # Step 4: Initialize camera
    camera = Camera(camera_id=CAMERA_ID)

    try:
        # Check whether the camera opened successfully
        if not camera.open():
            raise RuntimeError(
                f"Cannot open camera with ID {CAMERA_ID}"
            )

        print("Camera opened successfully")
        print("Press 'q' to quit")

        # Step 5: Real-time recognition loop
        while True:
            success, frame = camera.read_frame()

            if not success or frame is None:
                print("Cannot read camera frame")
                break

            # Step 6: Recognize faces
            start_time = time.perf_counter()
            
            results = recognizer.recognize(frame)
            
            processing_time = time.perf_counter() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0.0
            
            print(
                f"Recognition: {processing_time:.3f}s | "
                f"Processing FPS: {fps:.2f}"
                )

            # Step 7: Draw recognition results
            display_frame = draw_results(
                frame.copy(),
                results
            )

            cv2.imshow(
                "Real-time Face Recognition",
                display_frame
            )

            # Step 8: Exit when user presses q
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Always release resources
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()