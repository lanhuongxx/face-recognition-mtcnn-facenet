# MTCNN+ faceNet + Optical flow

import cv2
import time
from concurrent.futures import ThreadPoolExecutor

from src.camera import Camera
from src.config import (
    CAMERA_ID,
    REFERENCE_PATH,
    SIMILARITY_THRESHOLD,
)
from src.face_recognizer import FaceRecognizer
from src.face_tracker import OpticalFlowTracker


# Minimum interval between recognition requests.
RECOGNITION_INTERVAL = 2.0

# Stop showing an old identity after this duration.
LABEL_TIMEOUT = 5.0

# Maximum time between successful tracker updates.
# Tracking itself runs on every frame.


def recognize_frame(recognizer, frame):
    """Run face detection and recognition in background."""
    start = time.perf_counter()
    results = recognizer.recognize(frame)
    elapsed = time.perf_counter() - start
    return results, elapsed


def main():
    recognizer = FaceRecognizer(
        threshold=SIMILARITY_THRESHOLD,
    )

    reference = cv2.imread(REFERENCE_PATH)

    if reference is None:
        raise FileNotFoundError(
            f"Cannot read reference image: {REFERENCE_PATH}"
        )

    recognizer.register(reference)
    print("Reference registered successfully")

    camera = Camera(camera_id=CAMERA_ID)
    tracker = OpticalFlowTracker()

    if not camera.open():
        raise RuntimeError("Cannot open camera")

    recognition_future = None
    last_recognition_start = 0.0

    current_label = "Detecting..."
    current_similarity = None
    last_label_time = 0.0

    print("Automatic Face Recognition + Optical Flow")
    print("Press Q to quit")

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                success, frame = camera.read_frame()

                if not success or frame is None:
                    print("Cannot read camera frame")
                    break

                # 1. Track the current face on every frame.
                if tracker.is_tracking:
                    updated_box = tracker.update(frame)

                    if updated_box is None:
                        current_label = "Detecting..."
                        current_similarity = None
                        last_label_time = 0.0

                # 2. Collect completed recognition.
                if (
                    recognition_future is not None
                    and recognition_future.done()
                ):
                    results, elapsed = recognition_future.result()
                    recognition_future = None

                    print(
                        f"Recognition time: {elapsed:.3f}s | "
                        f"Faces: {len(results)}"
                    )

                    if results:
                        # This prototype handles one face only.
                        result = max(
                            results,
                            key=lambda item: item["similarity"],
                        )

                        detected_box = tuple(
                            map(int, result["box"])
                        )

                        if not tracker.is_tracking:
                            # Initialize from the detected box.
                            # This box belongs to an older frame.
                            initialized = tracker.initialize(
                                frame,
                                detected_box,
                            )

                            if initialized:
                                current_label = result["label"]
                                current_similarity = result["similarity"]
                                last_label_time = time.perf_counter()

                        else:
                            # Do not overwrite the current tracking box.
                            # Only accept identity when the old detection
                            # overlaps the current tracked region.
                            x1, y1, w1, h1 = tracker.box
                            x2, y2, w2, h2 = detected_box

                            ix1 = max(x1, x2)
                            iy1 = max(y1, y2)
                            ix2 = min(x1 + w1, x2 + w2)
                            iy2 = min(y1 + h1, y2 + h2)

                            intersection = (
                                max(0, ix2 - ix1)
                                * max(0, iy2 - iy1)
                            )

                            area1 = w1 * h1
                            area2 = w2 * h2

                            union = area1 + area2 - intersection
                            iou = intersection / union if union > 0 else 0.0

                            if iou >= 0.30:
                                current_label = result["label"]
                                current_similarity = result["similarity"]
                                last_label_time = time.perf_counter()

                    else:
                        # No face detected in the processed frame.
                        # Do not immediately reset a tracker that is
                        # still successfully following the current frame.
                        if not tracker.is_tracking:
                            current_label = "Detecting..."
                            current_similarity = None

                # 3. Expire stale recognition labels.
                if (
                    current_label in ("Matched", "Unknown")
                    and time.perf_counter() - last_label_time > LABEL_TIMEOUT
                ):
                    current_label = "Detecting..."
                    current_similarity = None

                # 4. Schedule recognition in background.
                now = time.perf_counter()

                if (
                    recognition_future is None
                    and now - last_recognition_start >= RECOGNITION_INTERVAL
                ):
                    recognition_future = executor.submit(
                        recognize_frame,
                        recognizer,
                        frame.copy(),
                    )
                    last_recognition_start = now

                # 5. Draw current tracking box.
                display = frame.copy()

                if tracker.is_tracking and tracker.box is not None:
                    x, y, w, h = tracker.box

                    cv2.rectangle(
                        display,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2,
                    )

                    if current_similarity is not None:
                        label_text = (
                            f"{current_label} "
                            f"({current_similarity:.2f})"
                        )
                    else:
                        label_text = current_label

                    cv2.putText(
                        display,
                        label_text,
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow(
                    "Face Recognition + Optical Flow Test",
                    display,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()