# Nhận diện khuôn mặt thời gian thực với MTCNN và FaceNet

Dự án Python được xây dựng theo kiến trúc mô-đun nhằm **xác minh khuôn mặt theo một ảnh tham chiếu trong thời gian thực** qua webcam. Hệ thống sử dụng MTCNN để phát hiện khuôn mặt, căn chỉnh theo vị trí hai mắt, trích xuất vector đặc trưng FaceNet 128 chiều thông qua DeepFace và so khớp bằng độ tương đồng cosine.

> **Phạm vi:** Đây là ứng dụng minh họa so khớp với **một danh tính tham chiếu**, không phải hệ thống nhận diện nhiều người từ cơ sở dữ liệu hay hệ thống xác thực đủ điều kiện triển khai thực tế.

## Chức năng chính

- Phát hiện khuôn mặt và năm điểm mốc trên khuôn mặt bằng **MTCNN**.
- Căn chỉnh khuôn mặt dựa trên vị trí hai mắt.
- Trích xuất **embedding 128 chiều** bằng **FaceNet** thông qua DeepFace.
- So khớp bằng **cosine similarity**, với ngưỡng cấu hình mặc định **0,70**.
- Hiển thị webcam theo thời gian thực, gồm khung khuôn mặt, điểm tương đồng và nhãn **Matched / Unknown**.
- Tách riêng các thành phần phát hiện, căn chỉnh, trích xuất đặc trưng, so khớp và trực quan hóa.
- Có các bài kiểm thử thành phần và notebook Jupyter để thực nghiệm, đánh giá.

## Quy trình nhận diện

```text
Ảnh tham chiếu ──► MTCNN ──► Căn chỉnh ──► FaceNet ──► Embedding tham chiếu
                                                              │
Khung hình webcam ► MTCNN ──► Căn chỉnh ──► FaceNet ──► Cosine similarity
                                                              │
                                                         Ngưỡng 0,70
                                                              │
                                                       Matched / Unknown
```

Độ tương đồng cosine giữa hai vector đặc trưng được tính theo công thức:

```text
similarity(A, B) = dot(A, B) / (||A||₂ × ||B||₂)
```

Nếu độ tương đồng với ảnh tham chiếu **lớn hơn hoặc bằng ngưỡng**, hệ thống trả về **Matched**; ngược lại là **Unknown**. Ngưỡng 0,70 là thông số của dự án, **chưa được hiệu chỉnh để đáp ứng yêu cầu bảo mật**.

## Cấu trúc dự án

```text
lab5_face_ID/
├── main.py                  # Điểm khởi chạy ứng dụng webcam
├── src/
│   ├── config.py            # Đường dẫn ảnh tham chiếu và ngưỡng so khớp
│   ├── camera.py            # Thu nhận hình ảnh từ webcam
│   ├── face_detector.py     # Phát hiện khuôn mặt và điểm mốc bằng MTCNN
│   ├── face_alignment.py    # Căn chỉnh theo hai mắt và cắt vùng khuôn mặt
│   ├── face_embedder.py     # Trích xuất embedding bằng FaceNet
│   ├── face_matcher.py      # Tính cosine similarity và đưa ra quyết định
│   ├── face_recognizer.py   # Kết hợp các bước trong pipeline
│   └── visualizer.py        # Vẽ khung khuôn mặt và nhãn dự đoán
├── data/
│   ├── reference/           # Ảnh tham chiếu cục bộ (không đưa lên Git)
│   └── evaluation/          # Ảnh đánh giá cục bộ (không đưa lên Git)
├── notebooks/               # Notebook thực nghiệm, đánh giá (nếu có)
├── document/                # Tài liệu/notebook (nếu được lưu tại đây)
├── scripts/
│   ├── checks/              # Kiểm tra thủ công từng thành phần
│   └── debug/               # Thử nghiệm và gỡ lỗi
├── tests/                   # Kiểm thử tự động
├── requirements.txt
└── README.md
```

> Cấu trúc trên mô tả các thư mục được sử dụng trong quá trình phát triển. Có thể điều chỉnh vị trí `notebooks/` và `document/` theo cấu trúc thực tế của repository.

## Yêu cầu môi trường

- **Python 3.13** (phiên bản sử dụng trong quá trình phát triển).
- Webcam hoạt động bình thường để chạy nhận diện trực tiếp.
- Các thư viện trong `requirements.txt`.
- Kết nối Internet ở lần khởi tạo FaceNet đầu tiên nếu cần tải trọng số mô hình đã huấn luyện.

Dự án được phát triển trên Windows với môi trường ảo Python. Khi chạy trên hệ điều hành khác, có thể cần điều chỉnh các phiên bản phụ thuộc, đặc biệt là TensorFlow.

## Hướng dẫn cài đặt

### 1. Sao chép repository

```powershell
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Thay URL mẫu bằng đường dẫn repository GitHub của bạn.

### 2. Tạo và kích hoạt môi trường ảo (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Cài đặt thư viện

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu `requirements.txt` chưa chứa các thư viện dùng cho notebook, cài bổ sung:

```powershell
python -m pip install matplotlib pandas ipykernel
```

**Lưu ý về phụ thuộc:** Cần sử dụng các phiên bản tương thích giữa DeepFace, TensorFlow, Keras và OpenCV. Nếu cài đặt trên môi trường mới gặp lỗi, nên cố định các phiên bản đã kiểm thử trong `requirements.txt`, đồng thời tránh cài chồng nhiều gói OpenCV khác nhau.

### 4. Chuẩn bị ảnh tham chiếu

Tạo thư mục `data/reference/` và thêm ảnh rõ nét, chỉ chứa **một khuôn mặt**. Với bộ ảnh đang dùng trong notebook, đường dẫn là:

```text
data/reference/image.jpg
```

Kiểm tra `REFERENCE_PATH` trong `src/config.py` và sửa thành `data/reference/image.jpg` nếu ứng dụng webcam vẫn trỏ tới tên cũ `my_face.jpg`.

**Quyền riêng tư:** Không đưa ảnh khuôn mặt cá nhân, khóa bí mật hoặc trọng số mô hình dung lượng lớn lên repository công khai. Cần cấu hình `.gitignore` phù hợp.

## Chạy ứng dụng webcam

Từ thư mục gốc dự án, sau khi kích hoạt môi trường ảo:

```powershell
python main.py
```

Ứng dụng sẽ đăng ký khuôn mặt từ ảnh tham chiếu, mở webcam mặc định và hiển thị kết quả so khớp. Nhấn **Q** trên cửa sổ webcam để thoát.

Nếu webcam không mở được, hãy kiểm tra quyền truy cập camera và bảo đảm không có ứng dụng khác đang sử dụng thiết bị.

## Chạy kiểm thử

```powershell
python -m pytest tests/ -v
```

Các bài kiểm thử hướng đến những thành phần như camera, phát hiện khuôn mặt, căn chỉnh, trích xuất embedding, so khớp, nhận diện và trực quan hóa. Một số bài kiểm thử có thể cần khởi tạo mô hình hoặc sử dụng dữ liệu cục bộ.

## Notebook thực nghiệm và đánh giá

Mở notebook bằng Jupyter hoặc VS Code và chọn Python kernel thuộc môi trường ảo của dự án. Nội dung thực nghiệm gồm:

1. Chuẩn bị môi trường và dữ liệu ảnh.
2. Phát hiện khuôn mặt bằng MTCNN và trực quan hóa các điểm mốc.
3. So sánh khuôn mặt trước và sau căn chỉnh.
4. Trích xuất embedding FaceNet 128 chiều.
5. So khớp bằng cosine similarity.
6. Thực nghiệm loại bỏ thành phần (*ablation study*): **có và không căn chỉnh khuôn mặt**.
7. Tổng hợp kết quả, đánh giá và thảo luận hạn chế.

Bộ ảnh cục bộ **hiện đang được sử dụng** trong notebook:

```text
data/reference/image.jpg
data/evaluation/image1.jpg
data/evaluation/image2.jpg
```

Các ảnh này không được phân phối kèm repository. Người chạy cần cung cấp ảnh của mình với sự đồng ý phù hợp; kết quả có thể khác nhau tùy dữ liệu.

## Kết quả thực nghiệm

> **Lưu ý:** Các số liệu dưới đây thuộc **bộ ảnh thực nghiệm trước khi thay dữ liệu** (một ảnh tham chiếu và hai ảnh kiểm tra cùng danh tính). Đây là nghiên cứu tình huống với ba ảnh, **không phải độ chính xác trên tập dữ liệu lớn**. Chưa thay thế các số liệu này bằng kết quả của `image.jpg`, `image1.jpg` và `image2.jpg`.

### Kết quả phát hiện khuôn mặt bằng MTCNN

| Ảnh | Số khuôn mặt phát hiện | Độ tin cậy phát hiện |
|---|---:|---:|
| Ảnh tham chiếu | 1 | 0,9988 |
| Ảnh bình thường | 1 | 0,9999 |
| Ảnh che mắt | 1 | 0,8155 |

### Kết quả so khớp và thực nghiệm căn chỉnh

| Điều kiện kiểm tra | Có căn chỉnh | Không căn chỉnh | Kết quả với ngưỡng 0,70 |
|---|---:|---:|---|
| Ảnh bình thường | 0,9765 | 0,9875 | Matched (cả hai) |
| Ảnh che mắt | 0,2971 | 0,2868 | Unknown (cả hai) |

**Nhận xét:** Ảnh bình thường được nhận diện đúng. Ảnh che mắt bị từ chối dù cùng danh tính với ảnh tham chiếu, tức xảy ra **từ chối sai (*false rejection*)**. Trong các mẫu thử này, căn chỉnh không làm tăng độ tương đồng một cách nhất quán và không thay đổi kết quả phân loại. Việc MTCNN phát hiện được khuôn mặt bị che mắt không bảo đảm bước so khớp danh tính sẽ thành công.

## Hạn chế

- Quy mô đánh giá còn nhỏ, chủ yếu với một danh tính tham chiếu.
- Ngưỡng **0,70** chưa được tối ưu trên một tập validation độc lập.
- Trường hợp ảnh che mắt bị phân loại **Unknown** là lỗi từ chối sai, **không phải** bằng chứng hệ thống nhận diện đúng người lạ.
- Ánh sáng, góc quay, chất lượng ảnh, che khuất và điều kiện webcam có thể ảnh hưởng đến độ tương đồng.
- So sánh có/không căn chỉnh chịu ảnh hưởng của cả thao tác xoay lẫn thay đổi vùng cắt khuôn mặt.
- Chưa có phát hiện khuôn mặt thật (*liveness detection*) hoặc chống giả mạo; **không phù hợp để sử dụng cho kiểm soát truy cập hay quyết định bảo mật quan trọng**.

## Hướng phát triển

- Đánh giá trên nhiều danh tính, bao gồm cặp ảnh cùng người và khác người.
- Tính tỷ lệ chấp nhận sai (**FAR**), tỷ lệ từ chối sai (**FRR**) và khảo sát ảnh hưởng của ngưỡng.
- Mở rộng thử nghiệm theo điều kiện ánh sáng, góc mặt, khoảng cách và che khuất.
- So sánh các phương pháp căn chỉnh và mô hình embedding khác nhau.
- Đo thời gian xử lý mỗi khung hình và tối ưu tốc độ nhận diện qua webcam.

## Công nghệ sử dụng

**Python · OpenCV · MTCNN · DeepFace / FaceNet · TensorFlow / Keras · NumPy · Matplotlib · Jupyter · pytest**

---

*Dự án học thuật thuộc lĩnh vực Thị giác máy tính. Chỉ sử dụng ảnh khuôn mặt khi có sự đồng ý phù hợp và tuân thủ các yêu cầu về quyền riêng tư.*