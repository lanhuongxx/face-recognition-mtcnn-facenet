
import cv2

from src.face_detector import FaceDetector


def main():
    image_path = "data/reference/person_01.jpg"

    image = cv2.imread(image_path)

    if image is None:
        print("Error: Cannot read image")
        return

    detector = FaceDetector()
    faces = detector.detect(image)

    print(f"Detected faces: {len(faces)}")

    for face in faces:
        print(face)

        x, y, w, h = face["box"]

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.imshow("MTCNN Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
