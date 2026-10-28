
import cv2

from src.face_recognizer import FaceRecognizer


def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {path}"
        )

    return image


def main():
    recognizer = FaceRecognizer(threshold=0.7)

    # 1. Đăng ký người cần nhận diện
    reference = load_image(
        "data/reference/person_01.jpg"
    )

    recognizer.register(reference)

    print("Reference registered successfully")

    # 2. Kiểm thử trên ảnh thật
    test_cases = [
        (
            "Same person",
            "data/reference/person_02.jpg",
            "Matched"
        ),
        (
            "Different person",
            "data/reference/stranger_01.jpg",
            "Unknown"
        )
    ]

    for name, path, expected in test_cases:
        image = load_image(path)

        results = recognizer.recognize(image)

        if len(results) != 1:
            print(
                f"{name}: Expected 1 face, "
                f"found {len(results)}"
            )
            continue

        result = results[0]

        actual = result["label"]
        similarity = result["similarity"]

        status = (
            "PASS" if actual == expected
            else "FAIL"
        )

        print(
            f"{name:17} | "
            f"Similarity: {similarity:.4f} | "
            f"Expected: {expected:7} | "
            f"Actual: {actual:7} | "
            f"{status}"
        )


if __name__ == "__main__":
    main()