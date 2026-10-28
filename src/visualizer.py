
# src/visualizer.py

import cv2


def draw_results(frame, results):
    """Draw bounding boxes and recognition labels."""
    height, width = frame.shape[:2]

    for result in results:
        x, y, w, h = result["box"]

        x1 = max(0, min(width, x))
        y1 = max(0, min(height, y))
        x2 = max(0, min(width, x + w))
        y2 = max(0, min(height, y + h))

        label = result["label"]
        similarity = result["similarity"]

        color = (
            (0, 255, 0)
            if label == "Matched"
            else (0, 0, 255)
        )

        if x2 <= x1 or y2 <= y1:
            continue

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        text = f"{label}: {similarity:.3f}"
        text_y = max(20, y1 - 10)

        cv2.putText(
            frame,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame