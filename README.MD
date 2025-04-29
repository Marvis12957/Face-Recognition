# Face Recognition Project

## Tổng quan
Dự án này xây dựng một hệ thống nhận diện khuôn mặt thời gian thực sử dụng webcam, dựa trên mô hình FaceNet và SVM. Các bước thực hiện bao gồm:

1. Thu thập ảnh từ webcam (thủ công hoặc tự động).
2. Cắt khuôn mặt từ ảnh gốc.
3. Finetune FaceNet để cải thiện độ chính xác.
4. Tạo embeddings từ khuôn mặt bằng FaceNet.
5. Vẽ biểu đồ phân bố embeddings (scatter plot).
6. Huấn luyện mô hình SVM để phân loại.
7. Nhận diện khuôn mặt thời gian thực.

**Mục tiêu**:
- Độ chính xác huấn luyện: ~90-95%.
- Độ chính xác thời gian thực: ~90-95%.

---

## Cấu trúc thư mục
- **`raw_images/`**: Chứa ảnh gốc thu thập từ webcam, phân loại theo tên người (ví dụ: `raw_images/hieu/`, `raw_images/dai/`).
- **`dataset/`**: Chứa ảnh khuôn mặt đã cắt, phân loại theo tên người (ví dụ: `dataset/hieu/`, `dataset/dai/`).
- **`model_pretrained/`**:
  - `yolov8s-face.pt`: Mô hình YOLOv8 để phát hiện khuôn mặt.
- **`embeddings.npy`**: File lưu embeddings của khuôn mặt.
- **`labels.npy`**: File lưu nhãn tương ứng với embeddings.
- **`svm_model.pkl`**: Mô hình SVM đã huấn luyện.
- **`facenet_finetuned.pth`**: Trọng số FaceNet sau khi finetune.
- **`embeddings_scatter_plot.png`**: Biểu đồ phân bố embeddings (tạo bởi `5_plot_embedding.py`).
- **`scripts/`**: Chứa các file mã nguồn:
  - `1_wedcam_capture_auto.py`: Chụp ảnh tự động từ webcam.
  - `1_wedcam_capture_manual.py`: Chụp ảnh thủ công từ webcam (nhấn Space).
  - `2_crop_faces.py`: Cắt khuôn mặt từ ảnh gốc.
  - `3_finetuning_facenet.py`: Finetune FaceNet.
  - `4_embedding.py`: Tạo embeddings từ khuôn mặt.
  - `5_plot_embedding.py`: Vẽ biểu đồ phân bố embeddings.
  - `6_train_svm.py`: Huấn luyện mô hình SVM.
  - `7_infer_svm.py`: Nhận diện khuôn mặt thời gian thực.
- **`README.MD`**: File hướng dẫn sử dụng.

---

## Yêu cầu cài đặt
Dự án sử dụng Python 3.8 trở lên. Cài đặt các thư viện sau:

```bash
pip install opencv-python
pip install torch torchvision
pip install facenet-pytorch
pip install ultralytics
pip install numpy
pip install scikit-learn
pip install joblib
pip install matplotlib
pip install seaborn
```

**Lưu ý**:
- Đảm bảo webcam hoạt động.
- Tải mô hình YOLOv8 face detection: `yolov8s-face.pt` (có thể tải từ [https://github.com/lindevs/yolov8-face?utm_source=chatgpt.com]).
- File `yolov8s-face.pt` đã được đặt trong thư mục `model_pretrained/`.

---

## Hướng dẫn sử dụng
### Bước 1: Thu thập ảnh từ webcam
#### Cách 1: Chụp thủ công (khuyến nghị để tăng tính đa dạng)
```bash
cd scripts
python 1_wedcam_capture_manual.py
```
- Nhập tên người (ví dụ: `hieu`).
- Nhấn **Space** để chụp ảnh khi khuôn mặt được phát hiện.
- Nhấn **ESC** để dừng.
- Ảnh được lưu vào `raw_images/{name}/`.

#### Cách 2: Chụp tự động
```bash
cd scripts
python 1_wedcam_capture_auto.py
```
- Nhập tên người (ví dụ: `hieu`).
- Chương trình tự động chụp khi phát hiện khuôn mặt, với khoảng cách 2 giây giữa các lần chụp.
- Nhấn **ESC** để dừng.
- Ảnh được lưu vào `raw_images/{name}/`.

**Lưu ý**:
- Thu thập 50 ảnh mỗi người.
- Thay đổi ánh sáng (tự nhiên, đèn trắng, đèn vàng), góc độ (nhìn thẳng, nghiêng 30-45°), biểu cảm (bình thường, cười), và chuyển động nhẹ để tăng tính đa dạng.

### Bước 2: Cắt khuôn mặt
```bash
cd scripts
python 2_crop_faces.py
```
- Đọc ảnh từ `raw_images/`.
- Cắt khuôn mặt bằng YOLOv8 và lưu vào `dataset/{name}/`.

### Bước 3: Finetune FaceNet
```bash
cd scripts
python 3_finetuning_facenet.py
```
- Finetune FaceNet trên dữ liệu trong `dataset/`.
- Trọng số tốt nhất được lưu vào `facenet_finetuned.pth`.

### Bước 4: Tạo embeddings
```bash
cd scripts
python 4_embedding.py
```
- Tạo embeddings từ ảnh trong `dataset/`.
- Lưu embeddings và nhãn vào `embeddings.npy` và `labels.npy`.

### Bước 5: (Tùy chọn) Vẽ biểu đồ phân bố embeddings
```bash
cd scripts
python 5_plot_embedding.py
```
- Vẽ scatter plot của embeddings bằng t-SNE.
- Lưu biểu đồ vào `embeddings_scatter_plot.png`.

### Bước 6: Huấn luyện SVM
```bash
cd scripts
python 6_train_svm.py
```
- Huấn luyện mô hình SVM trên embeddings.
- Lưu mô hình vào `svm_model.pkl`.

### Bước 7: Nhận diện khuôn mặt thời gian thực
```bash
cd scripts
python 7_infer_svm.py
```
- Sử dụng webcam để nhận diện khuôn mặt thời gian thực.
- Nhấn **ESC** để dừng.

---

## Lưu ý
1. **Đảm bảo dữ liệu đa dạng**:
   - Cosine similarity giữa các nhãn nên < 0.5.
   - Cosine similarity trong cùng nhãn nên < 0.9.
   - Xem output của `4_embedding.py` để kiểm tra.
2. **Số lượng ảnh**:
   - Nên có ít nhất 200 ảnh mỗi nhãn để finetuning và huấn luyện hiệu quả.
3. **Điều kiện ánh sáng**:
   - Khi nhận diện thời gian thực (`7_infer_svm.py`), đảm bảo ánh sáng tương tự như khi thu thập ảnh.
4. **Debugging**:
   - Nếu độ chính xác thấp, kiểm tra output của `4_embedding.py` (cosine similarity), `6_train_svm.py` (độ chính xác trên tập train/val/test), và `7_infer_svm.py` (xác suất dự đoán).
5. **Tối ưu thêm**:
   - Điều chỉnh `yolo.conf` trong `2_crop_faces.py` và `7_infer_svm.py` nếu phát hiện khuôn mặt không tốt (mặc định là 0.15).
   - Điều chỉnh ngưỡng xác suất trong `7_infer_svm.py` (mặc định là 0.75) để cân bằng giữa độ chính xác và tỷ lệ nhận diện "Unknown".

---

## Sinh viên thực hiện
Dự án được thực hiện bởi nhóm sinh viên sau đây. Bảng dưới đây phân chia công việc cụ thể của từng thành viên:

| Sinh viên              | Công việc                                                                 |
|------------------------|---------------------------------------------------------------------------|
| A48577 Trần Văn Hiếu   |                                                                           |
| A48451 Hoàng Minh Quân |                                                                           |
| A47272 Nguyễn Duy Đạt  |                                                                           |
---

## Liên hệ
Nếu bạn gặp vấn đề hoặc cần hỗ trợ thêm, hãy liên hệ với tôi qua MSTeams: A48577 - Trần Văn Hiếu.
