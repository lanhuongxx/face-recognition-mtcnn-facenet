
import cv2
import numpy as np


def align_face(
    image: np.ndarray,
    face: dict
) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    keypoints = face["keypoints"]

    left_eye = np.array(keypoints["left_eye"], dtype=np.float32)
    right_eye = np.array(keypoints["right_eye"], dtype=np.float32)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = np.degrees(np.arctan2(dy, dx))

    eye_center = (
        float((left_eye[0] + right_eye[0]) / 2),
        float((left_eye[1] + right_eye[1]) / 2)
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        eye_center,
        angle,
        1.0
    )

    height, width = image.shape[:2]

    aligned_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    x, y, w, h = face["box"]

    # Transform all four box corners before cropping
    corners = np.array(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ],
        dtype=np.float32
    ).reshape(1, 4, 2)

    rotated_corners = cv2.transform(
        corners,
        rotation_matrix
    )[0]

    x1 = max(0, int(np.floor(rotated_corners[:, 0].min())))
    y1 = max(0, int(np.floor(rotated_corners[:, 1].min())))
    x2 = min(width, int(np.ceil(rotated_corners[:, 0].max())))
    y2 = min(height, int(np.ceil(rotated_corners[:, 1].max())))

    aligned_crop = aligned_image[y1:y2, x1:x2]

    if aligned_crop.size == 0:
        raise ValueError("Aligned face crop is empty")

    return aligned_crop
