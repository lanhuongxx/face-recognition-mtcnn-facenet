
import cv2
from pathlib import Path

# Thư mục lưu ảnh
SAVE_DIR = Path("data/evaluation")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Mở webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

print("=== CAPTURE EVALUATION IMAGES ===")
print("Press 1: Normal face")
print("Press 2: Cover lower face")
print("Press 3: Cover eyes and upper nose")
print("Press Q: Quit")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read webcam frame")
            break

        # Hiển thị hướng dẫn trên bản xem trước
        preview = frame.copy()

        cv2.putText(
            preview,
            "1: Normal | 2: Lower | 3: Eyes | Q: Quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.imshow("Capture Evaluation", preview)

        key = cv2.waitKey(1) & 0xFF

        filenames = {
            ord("1"): "normal.jpg",
            ord("2"): "lower_occlusion.jpg",
            ord("3"): "eye_occlusion.jpg"
        }

        if key in filenames:
            path = SAVE_DIR / filenames[key]

            # Lưu frame gốc, không có chữ hướng dẫn
            cv2.imwrite(str(path), frame)

            print(f"Saved: {path}")

        elif key == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
