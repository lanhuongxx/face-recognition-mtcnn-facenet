# Nhận diện khuôn mặt thời gian thực — MTCNN, FaceNet và Optical Flow

Đồ án thị giác máy tính triển khai **xác minh khuôn mặt so với một ảnh tham chiếu** trên webcam. Hệ thống sử dụng MTCNN để phát hiện khuôn mặt và mốc mắt, căn chỉnh mặt, trích xuất embedding FaceNet 128 chiều (qua DeepFace), sau đó so sánh bằng cosine similarity. Lucas–Kanade Optical Flow hỗ trợ theo dõi giữa các lượt nhận diện bất đồng bộ.

> **Phạm vi hiện tại:** ứng dụng chính `main.py` chỉ chấp nhận kết quả nhận diện khi **có đúng một khuôn mặt**. Đây không phải hệ thống định danh nhiều người, không có kiểm tra người thật (liveness), và không phù hợp dùng trực tiếp cho kiểm soát ra vào.

## 1. Chức năng và nguyên lý

1. **Đăng ký ảnh tham chiếu:** đọc `data/reference/my_face.jpg`, phát hiện và trích xuất vector đặc trưng.
2. **Nhận diện bất đồng bộ:** định kỳ chạy MTCNN → alignment → FaceNet → cosine similarity trên ảnh chụp từ webcam.
3. **Ra quyết định:** `Matched` khi similarity **≥ 0.70**, ngược lại là `Unknown`. Ngưỡng là thiết lập của đồ án, chưa phải ngưỡng xác thực bảo mật đã hiệu chuẩn.
4. **Theo dõi:** Optical Flow Lucas–Kanade theo dõi khuôn mặt giữa các lần nhận diện; cơ chế lịch sử frame/replay giúp đồng bộ kết quả nhận diện đến trễ với frame hiện tại.
5. **Phục hồi:** có xử lý lỗi từ worker, neo lại tracker khi kết quả khớp, reset sau nhiều kết quả không hợp lệ hoặc IoU thấp; nếu khởi tạo tracker thất bại có thể hiển thị khung detection dự phòng rất ngắn (0,5 giây).

**Quy tắc hiển thị:** xanh lá = `Matched`; đỏ = `Unknown`; vàng = `Detecting...`. Khi hiện `[detection]`, đó là khung tạm, **không phải** kết quả đang được Optical Flow theo dõi.

**Xử lý nhiều khuôn mặt:** `main.py` **không chọn** khuôn mặt có similarity cao nhất. Nếu kết quả MTCNN/nhận diện có 0 hoặc từ 2 khuôn mặt trở lên, ứng dụng không cập nhật danh tính/tracker từ kết quả đó; khi kết quả hợp lệ cùng loại lặp lại đủ ngưỡng, tracker đang hoạt động có thể được reset. Benchmark độc lập có thể dùng cách cập nhật tracker khác và không đại diện toàn bộ logic `main.py`.

## 2. Cấu trúc thư mục

Cấu trúc các file chính được dùng (tên file notebook trong bản lưu của bạn có thể khác):

```text
lab5_face_ID/
├── main.py                              # Ứng dụng webcam chính (P3)
├── benchmark_webcam.py                  # Chương trình benchmark độc lập (nếu đã đổi tên)
│   hoặc benchmark_webcam_updated.py
├── requirements.txt
├── README.md
├── .gitignore
├── src/
│   ├── config.py
│   ├── camera.py
│   ├── face_detector.py
│   ├── face_alignment.py
│   ├── face_embedder.py
│   ├── face_matcher.py
│   ├── face_recognizer.py
│   ├── face_tracker.py                 # Bao gồm OpticalFlowTracker
│   ├── frame_sync.py                   # Đồng bộ frame và replay
│   └── visualizer.py                   # Tiện ích hiển thị (không bắt buộc được gọi từ main)
├── notebooks/
│   ├── 01_dataset_exploration_final.ipynb
│   └── 02_face_matching_visualization_reviewed_final.ipynb
├── data/
│   ├── gt_db/                          # Georgia Tech Face Database (chuẩn bị riêng)
│   ├── reference/                      # Ảnh tham chiếu riêng tư
│   └── evaluation/                     # Ảnh kiểm thử cá nhân (nếu có)
├── outputs/
│   ├── evaluation/                     # Kết quả đánh giá cặp ảnh
│   └── benchmark/                      # CSV benchmark
└── tests/                              # Kiểm thử tự động (nếu có)
```

**Chú ý:** tên notebook trên là tên đề xuất để lưu trong repo. Nếu file của bạn mang hậu tố `(2)` hoặc tên khác, dùng đúng tên file thực tế; không cần tạo thư mục `scripts/` hay khôi phục notebook cũ chỉ để khớp hướng dẫn.

## 3. Cài đặt

Môi trường đã dùng trong quá trình phát triển: **Windows, Python 3.13, `.venv`**, webcam và các thư viện trong `requirements.txt`.

```powershell
# Chạy tại thư mục gốc project
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

**Lưu ý về dependency:** danh sách phiên bản hiện tại được ghi từ môi trường phát triển; khả năng cài lại trên môi trường sạch **chưa được xác nhận**. Nếu `pip check` báo thiếu dependency của DeepFace, cần kiểm tra và chuẩn hóa `requirements.txt` ở bước tiếp theo. Không tự cài chồng các biến thể `opencv-python` và `opencv-*-headless` trong môi trường webcam đang hoạt động.

Lần đầu sử dụng FaceNet có thể cần kết nối mạng để tải trọng số mô hình.

## 4. Chuẩn bị dữ liệu và chạy ứng dụng

Đặt ảnh chứa **một khuôn mặt rõ nét** tại:

```text
data/reference/my_face.jpg
```

Đường dẫn và ngưỡng được cấu hình trong `src/config.py` (`REFERENCE_PATH`, `SIMILARITY_THRESHOLD`, `CAMERA_ID`).

Chạy:

```powershell
python main.py
```

Nhấn **Q** trong cửa sổ webcam để thoát. Console in thời gian nhận diện và bảng đếm lỗi/reset khi thoát.

## 5. Notebook thực nghiệm

Mở VS Code/Jupyter bằng kernel `.venv`, sau đó chạy lần lượt:

- **Notebook 01 — Dataset exploration:** thống kê Georgia Tech Face Database; trong lần chạy đã rà soát: 50 người, 750 ảnh.
- **Notebook 02 — Face matching visualization:** trực quan MTCNN, alignment, embedding FaceNet, similarity; thí nghiệm **A/B/C** (A = crop MTCNN thô; B = crop vuông + lề, không xoay; C = crop vuông + lề + xoay theo hai mắt); đánh giá 100 cặp ảnh; phân tích threshold; benchmark webcam.

Để chạy lại notebook cần chuẩn bị dữ liệu ở `data/gt_db/`, các ảnh tham chiếu/kiểm thử được notebook sử dụng và file CSV đánh giá/benchmark tương ứng. Notebook 02 có phần xác định `PROJECT_DIR` để chỉ đến thư mục project nếu working directory của Jupyter khác nơi chứa mã nguồn.

**Diễn giải:** so sánh B → C giúp xem tác động của phép xoay khi giữ cách crop nhất quán hơn; kết quả trên ít ảnh không đủ chứng minh alignment luôn có lợi. Độ chính xác 100% trên 100 cặp thử **không** đồng nghĩa độ chính xác 100% ngoài thực tế.

## 6. Benchmark webcam

Nếu file đang là `benchmark_webcam_updated.py`:

```powershell
python benchmark_webcam_updated.py
```

Nếu đã đặt tên là `benchmark_webcam.py`, thay bằng `python benchmark_webcam.py`.

Kết quả CSV được ghi vào `outputs/benchmark/` (ví dụ `summary.csv`, `tracking_latency.csv`, `recognition_latency.csv`).

**Phiên đo gần nhất được báo cáo:** 60,02 giây, 1.794 frame, FPS vòng lặp hiển thị trung bình **29,89**, tracking latency trung bình **1,56 ms** (1.778 mẫu), recognition latency trung bình **401,22 ms** (30 mẫu). Đây là **benchmark độc lập**, không phải phép đo toàn bộ logic replay/reanchor/fallback trong `main.py` P3. FPS hiển thị không đồng nghĩa số lượt FaceNet nhận diện mỗi giây.

## 7. Kiểm thử

Nếu có thư mục `tests/`:

```powershell
python -m pytest tests/ -v
```

Nên thử webcam lần lượt: mặt tham chiếu, mặt khác, rời khung rồi quay lại, chuyển động nhanh, một phần khuôn mặt bị che, và thoát bằng Q. Việc bị che có thể làm MTCNN không phát hiện mặt hoặc FaceNet trả về `Unknown`; đây là giới hạn cần nêu, không tự động là lỗi chương trình.

## 8. Giới hạn và bảo mật dữ liệu

- Hiện chỉ xử lý **một khuôn mặt** trong ứng dụng chính; nhiều khuôn mặt có thể làm tracker bị reset.
- MTCNN/FaceNet nhạy với điều kiện ánh sáng, góc nhìn, che khuất và chất lượng ảnh.
- Ngưỡng 0,70 chưa được hiệu chuẩn từ tập validation độc lập đủ lớn.
- Benchmark và ứng dụng chính có thể khác logic tracking, vì vậy không dùng FPS benchmark để khẳng định đầy đủ độ chính xác tracker ứng dụng chính.
- Lịch sử frame và thao tác copy ảnh có thể tăng RAM khi dùng webcam độ phân giải cao.
- Không có chống giả mạo (anti-spoofing) hoặc xác thực người thật.
- **Không đưa `data/reference/my_face.jpg` hay ảnh riêng tư lên GitHub công khai.** Kiểm tra cả file đã được Git track trước khi push; `.gitignore` không tự xóa file khỏi lịch sử Git.

## 9. Thành phần công nghệ

Python · OpenCV · MTCNN · DeepFace/FaceNet · TensorFlow/tf-keras · NumPy · Jupyter · Pandas · Matplotlib · pytest.

---

*Tài liệu mô tả phiên bản ứng dụng một khuôn mặt và các thí nghiệm đã thực hiện; không tuyên bố hệ thống đã sẵn sàng triển khai bảo mật thực tế.*