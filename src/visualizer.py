import cv2


def draw_results(frame, results):
    """Draw bounding boxes and recognition labels."""

    if frame is None or frame.size == 0:
        raise ValueError("Frame is empty")

    height, width = frame.shape[:2]

    for result in results:
        x, y, w, h = result["box"]

        # Clip bounding box to image boundaries
        x1 = max(0, min(width - 1, int(x)))
        y1 = max(0, min(height - 1, int(y)))
        x2 = max(0, min(width - 1, int(x + w)))
        y2 = max(0, min(height - 1, int(y + h)))

        if x2 <= x1 or y2 <= y1:
            continue

        label = result["label"]
        similarity = result["similarity"]

        color = (
            (0, 255, 0)
            if label == "Matched"
            else (0, 0, 255)
        )

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        # Prepare recognition label
        text = f"{label}: {similarity:.3f}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        # Prevent text from exceeding the right boundary
        text_x = max(
            0,
            min(x1, width - text_width)
        )

        # Prefer above the face, otherwise below its top edge
        if y1 >= text_height + baseline + 10:
            text_y = y1 - 10
        else:
            text_y = min(
                height - baseline - 1,
                y1 + text_height + 10
            )

        # Draw recognition text
        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            font_scale,
            color,
            thickness
        )

    return frame