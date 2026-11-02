# Nhận diện khuôn mặt thời gian thực

Dự án minh họa cách xác minh khuôn mặt qua webcam bằng **MTCNN**, **FaceNet** và **OpenCV**. Chương trình so sánh khuôn mặt trước camera với **một ảnh tham chiếu** do bạn cung cấp, sau đó hiển thị nhãn `Matched` hoặc `Unknown`.

> Đây là project học tập/thực nghiệm, không phải hệ thống xác thực an toàn hay công cụ nhận diện nhiều người.

## Chương trình hoạt động như thế nào?

```text
Ảnh tham chiếu ──► MTCNN ──► Căn chỉnh khuôn mặt ──► FaceNet ──► Embedding tham chiếu

Webcam ──► Nhận diện khuôn mặt ──► FaceNet ──► Cosine similarity
                                                        │
                                        So với ngưỡng 0,70
                                                        │
                                                Matched / Unknown

Các frame webcam ──► Optical Flow ──► Theo dõi khung khuôn mặt
```

Các bước chính:

1. **MTCNN** tìm khuôn mặt và các điểm mốc trên ảnh.
2. Chương trình căn chỉnh khuôn mặt dựa trên vị trí hai mắt.
3. **FaceNet** trích xuất embedding 128 chiều qua thư viện DeepFace.
4. **Cosine similarity** so sánh embedding webcam với embedding tham chiếu. Điểm từ `0,70` trở lên được gán nhãn `Matched`.
5. **Lucas–Kanade Optical Flow** theo dõi khung khuôn mặt giữa các lần nhận diện.

Nhận diện chạy bất đồng bộ, tối đa khoảng **một lần mỗi 2 giây**; theo dõi cập nhật trên các frame webcam. Nhãn nhận diện cũ hết hạn sau **5 giây**. Nếu có nhiều khuôn mặt, ứng dụng hiện chọn kết quả có similarity cao nhất và chỉ theo dõi một khuôn mặt.

## Cấu trúc dự án

```text
lab5_face_ID/
├── main.py                              # Chạy ứng dụng webcam
├── benchmark_webcam.py                  # Đo hiệu năng webcam và ghi CSV
├── src/
│   ├── config.py                        # Ảnh tham chiếu, ngưỡng, camera
│   ├── camera.py                        # Mở, đọc và giải phóng camera
│   ├── face_detector.py                 # Phát hiện khuôn mặt bằng MTCNN
│   ├── mtcnn_compat.py                  # Hỗ trợ tương thích MTCNN
│   ├── face_alignment.py                # Căn chỉnh khuôn mặt theo mắt
│   ├── face_embedder.py                 # Trích xuất embedding FaceNet
│   ├── face_matcher.py                  # Tính cosine similarity
│   ├── face_recognizer.py               # Kết nối các bước nhận diện
│   ├── face_tracker.py                  # Theo dõi bằng Optical Flow / MIL
│   └── visualizer.py                    # Vẽ kết quả lên hình
├── data/
│   ├── reference/                       # Ảnh dùng để đăng ký khuôn mặt
│   ├── evaluation/                      # Ảnh thực nghiệm cục bộ
│   ├── gt_db/                           # Bộ dữ liệu Georgia Tech cục bộ
│   └── pair_evaluation/pairs.csv        # Các cặp ảnh dùng để đánh giá
├── document/
│   ├── 01_dataset_exploration_final.ipynb
│   └── 02_face_matching_visualization_final.ipynb
├── outputs/
│   ├── evaluation/pair_results.csv
│   └── benchmark/                       # CSV kết quả benchmark
├── scripts/
│   ├── checks/                          # Kiểm tra từng thành phần
│   └── debug/                           # Script thử nghiệm/gỡ lỗi
├── tests/                               # Unit tests
├── requirements.txt
└── .gitignore
```

## Cài đặt trên Windows

Mở PowerShell tại thư mục gốc dự án, sau đó tạo và kích hoạt môi trường ảo:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cài các thư viện và kiểm tra dependency:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

`requirements.txt` đang ghim các phiên bản thư viện dùng cho môi trường project. TensorFlow, OpenCV và các thư viện liên quan có thể phụ thuộc hệ điều hành/Python; nếu cài trên môi trường khác, hãy kiểm tra tính tương thích của các phiên bản.

Ở lần chạy đầu, thư viện có thể cần tải trọng số mô hình đã huấn luyện. Việc này cần kết nối Internet.

## Chuẩn bị ảnh tham chiếu

Đặt ảnh rõ nét, chứa **đúng một khuôn mặt** tại:

```text
data/reference/my_face.jpg
```

Ứng dụng lấy đường dẫn này từ `REFERENCE_PATH` trong `src/config.py`. Nếu muốn dùng file khác, hãy cập nhật cấu hình tương ứng.

Ảnh trong `data/reference/`, `data/evaluation/` và dữ liệu thử nghiệm là dữ liệu cục bộ, có thể không được đưa lên Git. Hãy chỉ sử dụng ảnh khuôn mặt khi có sự đồng ý phù hợp và không công khai dữ liệu cá nhân.

## Chạy ứng dụng webcam

Đảm bảo đã kết nối webcam, đặt ảnh tham chiếu đúng vị trí, rồi chạy từ thư mục gốc:

```powershell
python main.py
```

Ứng dụng đăng ký khuôn mặt từ ảnh tham chiếu, mở webcam và hiển thị khung theo dõi cùng nhãn nhận diện. Nhấn **Q** trên cửa sổ webcam để thoát. Nếu không mở được camera, hãy kiểm tra quyền truy cập camera và đóng các ứng dụng khác đang sử dụng thiết bị.

Các giá trị chính có thể điều chỉnh trong `src/config.py`:

| Cấu hình | Mặc định | Ý nghĩa |
|---|---:|---|
| `REFERENCE_PATH` | `data/reference/my_face.jpg` | Ảnh đăng ký khuôn mặt |
| `SIMILARITY_THRESHOLD` | `0.70` | Ngưỡng phân loại `Matched` |
| `CAMERA_ID` | `0` | Chỉ số webcam |
| `MIN_FACE_SIZE` | `50` | Kích thước khuôn mặt tối thiểu (pixel) |
| `MIN_CONFIDENCE` | `0.90` | Độ tin cậy phát hiện tối thiểu |

Ngưỡng `0,70` là giá trị dùng cho bài thực nghiệm, không được hiệu chỉnh để bảo đảm an toàn hoặc phù hợp cho mọi người và điều kiện camera.

## Chạy kiểm thử

```powershell
python -m pytest tests/ -v
```

Tests bao phủ các thành phần camera, detector, alignment, embedder, matcher, recognizer, tracker và visualizer. Phần lớn là unit test; kết quả test không thay thế cho việc đánh giá độ chính xác trên dữ liệu thực tế.

## Đo hiệu năng webcam

```powershell
python benchmark_webcam.py
```

Benchmark chạy tối đa khoảng **60 giây**, có thể nhấn **Q** để dừng sớm. Giữ một khuôn mặt trong khung hình và di chuyển tự nhiên để đo. Kết quả được lưu ở:

```text
outputs/benchmark/summary.csv
outputs/benchmark/tracking_latency.csv
outputs/benchmark/recognition_latency.csv
```

FPS trong báo cáo là tốc độ vòng lặp hiển thị, không phải tốc độ suy luận FaceNet. Chương trình benchmark cũng cần ảnh tham chiếu và webcam hoạt động.

## Notebook và dữ liệu đánh giá

Trong thư mục `document/`:

- **`01_dataset_exploration_final.ipynb`**: khám phá bộ Georgia Tech Face Database trong `data/gt_db/` và kiểm tra các cặp ảnh đánh giá.
- **`02_face_matching_visualization_final.ipynb`**: minh họa MTCNN, căn chỉnh, FaceNet, cosine similarity; so sánh có/không căn chỉnh và trực quan hóa kết quả đánh giá/benchmark.

Notebook sử dụng dữ liệu cục bộ như `data/gt_db/`, `data/pair_evaluation/pairs.csv`, ảnh trong `data/reference/` và `data/evaluation/`, cùng một số CSV trong `outputs/`. Các dữ liệu này có thể không được đóng gói cùng repository. Hãy kiểm tra các đường dẫn trong notebook và chuẩn bị đúng dữ liệu trước khi chạy; nên chạy notebook khám phá dữ liệu trước notebook trực quan hóa.

Ảnh tham chiếu của notebook có thể khác ảnh mà ứng dụng webcam sử dụng. Nếu cần so sánh kết quả giữa notebook và webcam, hãy đảm bảo cả hai dùng cùng một ảnh tham chiếu.

## Giới hạn và lưu ý

- Ứng dụng xác minh với **một ảnh tham chiếu**, không quản lý cơ sở dữ liệu danh tính.
- Tracking chỉ theo dõi một khuôn mặt. Nếu khuôn mặt di chuyển nhanh, kết quả nhận diện từ frame cũ có thể không còn khớp hoàn toàn với vị trí hiện tại.
- Độ sáng, góc mặt, che khuất và kích thước khuôn mặt có thể ảnh hưởng đến kết quả.
- Project không có phát hiện giả mạo/liveness detection; ảnh hoặc video có thể đánh lừa hệ thống.
- Điểm similarity và kết quả thử nghiệm không chứng minh độ chính xác trong mọi tình huống.

**Chỉ sử dụng cho mục đích học tập và thử nghiệm; không dùng làm cơ chế kiểm soát truy cập hoặc quyết định an toàn quan trọng.**