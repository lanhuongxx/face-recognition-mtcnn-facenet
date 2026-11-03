"""Single-face recognition with asynchronous MTCNN/FaceNet and optical-flow tracking."""

import time
from concurrent.futures import ThreadPoolExecutor

import cv2

from src.camera import Camera
from src.config import CAMERA_ID, REFERENCE_PATH, SIMILARITY_THRESHOLD
from src.face_recognizer import FaceRecognizer
from src.face_tracker import OpticalFlowTracker
from src.frame_sync import (
    FrameHistory,
    calculate_iou,
    is_recognition_valid,
    propagate_box,
    replay_snapshot,
)

RECOGNITION_INTERVAL = 2.0
LABEL_TIMEOUT = 5.0
MAX_RESULT_AGE = 3.0
MAX_CATCHUP_FRAMES = 3
MIN_ASSOCIATION_IOU = 0.30
HISTORY_FRAMES = 90
NO_FACE_RESET_STREAK = 2
MISMATCH_RESET_STREAK = 3
# A detection-only box is valid only briefly; it is not a tracker.
FALLBACK_BOX_TTL = 0.5  # seconds


def recognize_frame(recognizer, frame):
    """Recognize faces on an immutable snapshot in a background worker."""
    start = time.perf_counter()
    results = recognizer.recognize(frame)
    return results, time.perf_counter() - start


def reset_label():
    return "Detecting...", None, 0.0


def try_initialize(frame, box):
    """Initialize a replacement tracker without disturbing the live one."""
    candidate = OpticalFlowTracker()
    try:
        initialized = candidate.initialize(frame, box)
    except ValueError as exc:
        print(f"[Initialize error] {exc}")
        return None
    return candidate if initialized else None


def main():
    recognizer = FaceRecognizer(threshold=SIMILARITY_THRESHOLD)
    reference = cv2.imread(REFERENCE_PATH)
    if reference is None:
        raise FileNotFoundError(f"Cannot read reference image: {REFERENCE_PATH}")
    recognizer.register(reference)
    print("Reference registered successfully")

    camera = Camera(camera_id=CAMERA_ID)
    tracker = OpticalFlowTracker()
    history = FrameHistory(max_frames=HISTORY_FRAMES)
    if not camera.open():
        raise RuntimeError("Cannot open camera")

    recognition_future = None
    recognition_job = None
    replay_future = None
    replay_job = None
    frame_id = 0
    track_generation = 0
    last_recognition_start = 0.0
    current_label, current_similarity, last_label_time = reset_label()
    recognition_errors = 0
    replay_errors = 0
    replay_rejected = 0
    iou_mismatches = 0
    reanchor_failures = 0
    no_face_streak = 0
    mismatch_streak = 0
    no_face_resets = 0
    mismatch_resets = 0
    init_failures = 0
    fallback_activations = 0
    fallback_box = None
    fallback_expires_at = 0.0

    print("Automatic Face Recognition + Optical Flow")
    print("Press Q to quit")

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            while True:
                success, frame = camera.read_frame()
                if not success or frame is None:
                    print("Cannot read camera frame")
                    break

                frame_id += 1
                now = time.perf_counter()
                history.append(frame_id, frame)

                # Update the live tracker on every camera frame.
                if tracker.is_tracking and tracker.update(frame) is None:
                    track_generation += 1
                    current_label, current_similarity, last_label_time = reset_label()
                    no_face_streak = 0
                    mismatch_streak = 0
                    fallback_box = None

                # Recognition result belongs to its submitted source frame.
                if recognition_future is not None and recognition_future.done():
                    job = recognition_job
                    try:
                        results, elapsed = recognition_future.result()
                    except Exception as exc:
                        recognition_errors += 1
                        print(f"[Recognition error #{recognition_errors}] {type(exc).__name__}: {exc}")
                        results = None
                    finally:
                        recognition_future = None
                        recognition_job = None
                    if results is not None:
                        print(f"Recognition time: {elapsed:.3f}s | Faces: {len(results)}")

                    recognition_valid = (
                        results is not None
                        and is_recognition_valid(job, track_generation, now, MAX_RESULT_AGE)
                    )
                    if recognition_valid and len(results) != 1:
                        # Only current-generation results can affect the live tracker.
                        no_face_streak += 1
                        mismatch_streak = 0
                        print(f"[Face count] {len(results)} face(s); streak={no_face_streak}")
                        if no_face_streak >= NO_FACE_RESET_STREAK:
                            if tracker.is_tracking:
                                tracker.reset()
                                track_generation += 1
                                no_face_resets += 1
                                print(f"[Tracker reset] invalid face count (#{no_face_resets})")
                            current_label, current_similarity, last_label_time = reset_label()
                            fallback_box = None
                            no_face_streak = 0
                    elif recognition_valid and len(results) == 1:
                        no_face_streak = 0

                    if (
                        recognition_valid
                        and replay_future is None
                        and len(results) == 1
                    ):
                        result = results[0]
                        source_box = tuple(map(int, result["box"]))
                        frames = history.get_from(job["frame_id"])
                        if frames is not None:
                            replay_job = {
                                "submitted_at": job["submitted_at"],
                                "generation": job["generation"],
                                "result": result,
                            }
                            replay_future = executor.submit(
                                replay_snapshot,
                                list(frames),
                                job["frame_id"],
                                source_box,
                            )

                # Replay runs in the background; only a tiny catch-up is live.
                if replay_future is not None and replay_future.done():
                    job = replay_job
                    replay_failed = False
                    try:
                        replay_result = replay_future.result()
                    except Exception as exc:
                        replay_errors += 1
                        replay_failed = True
                        print(f"[Replay error #{replay_errors}] {type(exc).__name__}: {exc}")
                        replay_result = None
                    finally:
                        replay_future = None
                        replay_job = None

                    # Errors and rejected results are separate metrics.
                    if not replay_failed:
                        rejection_reason = None
                        if replay_result is None:
                            rejection_reason = "replay returned no result"
                        elif not is_recognition_valid(
                            job, track_generation, now, MAX_RESULT_AGE
                        ):
                            rejection_reason = "stale result or tracking generation changed"
                        else:
                            replay_frame_id, replay_box = replay_result
                            remaining = history.get_from(replay_frame_id)
                            if remaining is None:
                                rejection_reason = "replay frame not in history"
                            elif len(remaining) - 1 > MAX_CATCHUP_FRAMES:
                                rejection_reason = (
                                    f"catch-up exceeds {MAX_CATCHUP_FRAMES} frames "
                                    f"({len(remaining) - 1} pending)"
                                )
                            elif remaining[-1][0] != frame_id:
                                rejection_reason = "history does not reach current frame"
                            else:
                                catchup = FrameHistory(max_frames=max(2, len(remaining)))
                                for saved_id, saved_frame in remaining:
                                    catchup.append(saved_id, saved_frame)
                                propagated = propagate_box(
                                    catchup, replay_frame_id, replay_box
                                )
                                if propagated is None or propagated[0] != frame_id:
                                    rejection_reason = "catch-up propagation failed"
                                else:
                                    current_box = propagated[1]

                        if rejection_reason is not None:
                            replay_rejected += 1
                            print(
                                f"[Replay rejected #{replay_rejected}] "
                                f"{rejection_reason}"
                            )
                        else:
                            result = job["result"]
                            if tracker.is_tracking:
                                iou = calculate_iou(tracker.box, current_box)
                                if iou >= MIN_ASSOCIATION_IOU:
                                    candidate = try_initialize(frame, current_box)
                                    if candidate is not None:
                                        tracker = candidate
                                        fallback_box = None
                                        track_generation += 1
                                    else:
                                        reanchor_failures += 1
                                        print(
                                            f"[Reanchor failed #{reanchor_failures}] "
                                            "keeping previous tracker"
                                        )
                                    accepted = True
                                    mismatch_streak = 0
                                else:
                                    accepted = False
                                    iou_mismatches += 1
                                    mismatch_streak += 1
                                    print(
                                        f"[IoU mismatch #{iou_mismatches}] "
                                        f"iou={iou:.2f}, minimum={MIN_ASSOCIATION_IOU:.2f}, "
                                        f"streak={mismatch_streak}"
                                    )
                                    if mismatch_streak >= MISMATCH_RESET_STREAK:
                                        tracker.reset()
                                        track_generation += 1
                                        current_label, current_similarity, last_label_time = reset_label()
                                        mismatch_resets += 1
                                        mismatch_streak = 0
                                        fallback_box = None
                                        print(f"[Tracker reset] repeated IoU mismatch (#{mismatch_resets})")
                            else:
                                candidate = try_initialize(frame, current_box)
                                accepted = candidate is not None
                                if accepted:
                                    tracker = candidate
                                    fallback_box = None
                                    track_generation += 1
                                    mismatch_streak = 0
                                else:
                                    init_failures += 1
                                    fallback_activations += 1
                                    # current_box was propagated to the *current* camera frame.
                                    # Never reuse this static box after its short TTL.
                                    fallback_box = tuple(map(int, current_box))
                                    fallback_expires_at = now + FALLBACK_BOX_TTL
                                    accepted = True  # Face recognized; only LK tracking failed.
                                    print(
                                        f"[Initialize failed #{init_failures}] "
                                        f"using detection fallback for {FALLBACK_BOX_TTL:.1f}s"
                                    )

                            if accepted:
                                current_label = result["label"]
                                current_similarity = result["similarity"]
                                last_label_time = now

                if (
                    current_label in ("Matched", "Unknown")
                    and now - last_label_time > LABEL_TIMEOUT
                ):
                    current_label, current_similarity, last_label_time = reset_label()

                # Only one recognition/replay pipeline at a time.
                if (
                    recognition_future is None
                    and replay_future is None
                    and now - last_recognition_start >= RECOGNITION_INTERVAL
                ):
                    recognition_job = {
                        "frame_id": frame_id,
                        "submitted_at": now,
                        "generation": track_generation,
                    }
                    recognition_future = executor.submit(
                        recognize_frame, recognizer, frame.copy()
                    )
                    last_recognition_start = now

                # Never present an old static detection as live optical-flow tracking.
                if fallback_box is not None and now >= fallback_expires_at:
                    fallback_box = None

                display = frame.copy()
                draw_box = (
                    tracker.box if tracker.is_tracking and tracker.box is not None
                    else fallback_box
                )
                if draw_box is not None:
                    x, y, w, h = draw_box
                    box_color = (0, 255, 0) if current_label == "Matched" else (
                        (0, 0, 255) if current_label == "Unknown" else (0, 255, 255)
                    )
                    # Preserve Matched/Unknown/Detecting colors for fallback as well.
                    cv2.rectangle(display, (x, y), (x + w, y + h), box_color, 2)
                    label_text = current_label
                    if not tracker.is_tracking:
                        label_text += " [detection]"
                    if current_similarity is not None:
                        label_text += f" ({current_similarity:.2f})"
                    cv2.putText(
                        display,
                        label_text,
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_color,
                        2,
                    )

                cv2.imshow("Face Recognition + Optical Flow Test", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        print(
            f"Summary: recognition_errors={recognition_errors}, "
            f"replay_errors={replay_errors}, replay_rejected={replay_rejected}, "
            f"iou_mismatches={iou_mismatches}, reanchor_failures={reanchor_failures}, "
            f"no_face_resets={no_face_resets}, mismatch_resets={mismatch_resets}, "
            f"init_failures={init_failures}, fallback_activations={fallback_activations}"
        )
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()