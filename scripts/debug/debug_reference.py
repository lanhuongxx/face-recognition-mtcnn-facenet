
import cv2

from src.face_detector import FaceDetector


REFERENCE_PATH = "data/reference/my_face.jpg"


def main():
    image = cv2.imread(REFERENCE_PATH)

    if image is None:
        raise FileNotFoundError(REFERENCE_PATH)

    detector = FaceDetector()
    faces = detector.detect(image)

    MIN_FACE_SIZE = 50
    valid_faces = [
        
        face
        for face in faces
        if face["box"][2] >= MIN_FACE_SIZE
        and face["box"][3] >= MIN_FACE_SIZE
        ]
    
    print(f"Before filtering: {len(faces)}")
    print(f"After filtering: {len(valid_faces)}")
    
    for face in valid_faces:
        print(
            f"Valid face: {face['box']}, "
            f"confidence={face['confidence']:.4f}"
            )

    print(f"Detected faces: {len(faces)}")

    for index, face in enumerate(faces):
        x, y, w, h = face["box"]
        confidence = face["confidence"]

        print(
            f"Face {index + 1}: "
            f"box={face['box']}, "
            f"confidence={confidence:.4f}"
        )

        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            f"Face {index + 1}: {confidence:.2f}",
            (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Reference Detection Debug", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
