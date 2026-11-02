import cv2
import numpy as np


def align_face(
    image: np.ndarray,
    face: dict
) -> np.ndarray:
    """Căn chỉnh khuôn mặt theo hai mắt và trả về crop 160x160."""

    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel image")

    keypoints = face["keypoints"]

    left_eye = np.array(
        keypoints["left_eye"], dtype=np.float32
    )
    right_eye = np.array(
        keypoints["right_eye"], dtype=np.float32
    )

    if (
        left_eye.shape != (2,)
        or right_eye.shape != (2,)
        or not np.isfinite(left_eye).all()
        or not np.isfinite(right_eye).all()
    ):
        raise ValueError("Invalid eye landmarks")

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    if np.hypot(dx, dy) < 1:
        raise ValueError("Eye landmarks are too close")

    angle = np.degrees(np.arctan2(dy, dx))

    eye_center = (
        float((left_eye[0] + right_eye[0]) / 2),
        float((left_eye[1] + right_eye[1]) / 2),
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        eye_center,
        angle,
        1.0,
    )

    height, width = image.shape[:2]

    aligned_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    x, y, w, h = face["box"]

    if w <= 0 or h <= 0:
        raise ValueError("Invalid face bounding box")

    # Transform the center of the original bounding box
    center = np.array(
        [[[x + w / 2, y + h / 2]]],
        dtype=np.float32,
    )

    rotated_center = cv2.transform(
        center,
        rotation_matrix,
    )[0, 0]

    cx, cy = rotated_center

    # Square crop with 25% margin
    side = max(1, int(np.ceil(max(w, h) * 1.25)))

    # Padding prevents an empty crop near image borders
    pad = side

    padded_image = cv2.copyMakeBorder(
        aligned_image,
        pad, pad, pad, pad,
        cv2.BORDER_REFLECT_101,
    )

    x1 = int(round(cx - side / 2)) + pad
    y1 = int(round(cy - side / 2)) + pad

    aligned_crop = padded_image[
        y1:y1 + side,
        x1:x1 + side,
    ]

    if (
        aligned_crop.size == 0
        or aligned_crop.shape[0] != side
        or aligned_crop.shape[1] != side
    ):
        raise ValueError("Aligned face crop is invalid")

    aligned_crop = cv2.resize(
        aligned_crop,
        (160, 160),
        interpolation=cv2.INTER_LINEAR,
    )

    return aligned_crop