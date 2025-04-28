from ultralytics import YOLO
import cv2
import os
import numpy as np

# Tải mô hình YOLOv8
yolo = YOLO("yolov8s-face.pt")
yolo.conf = 0.15  # Giảm ngưỡng để tăng khả năng phát hiện

INPUT_DIR = 'raw_images'
OUTPUT_DIR = 'dataset'

os.makedirs(OUTPUT_DIR, exist_ok=True)

for person in os.listdir(INPUT_DIR):
    person_folder = os.path.join(INPUT_DIR, person)
    if not os.path.isdir(person_folder):
        continue

    out_person_dir = os.path.join(OUTPUT_DIR, person)
    os.makedirs(out_person_dir, exist_ok=True)

    img_count = 0

    for fname in os.listdir(person_folder):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        img_path = os.path.join(person_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Lỗi đọc ảnh: {img_path}")
            continue

        # Nhận diện khuôn mặt
        results = yolo(img)
        detections = results[0].boxes.xyxy.numpy()
        confidences = results[0].boxes.conf.numpy()
        classes = results[0].boxes.cls.numpy()

        if len(detections) > 0:
            # Lấy khuôn mặt có conf cao nhất
            max_conf_idx = np.argmax(confidences)
            x1, y1, x2, y2 = detections[max_conf_idx]
            conf = confidences[max_conf_idx]
            cls = classes[max_conf_idx]
            if int(cls) != 0:  # Chỉ xử lý khuôn mặt (class 0)
                continue
            if (x2 - x1) < 50 or (y2 - y1) < 50:
                continue

            # Cắt khuôn mặt
            try:
                face = img[int(y1):int(y2), int(x1):int(x2)]
                if face.size == 0:
                    continue
                if face.shape[0] < 50 or face.shape[1] < 50:
                    print(f"Khuôn mặt sau khi cắt quá nhỏ: {fname}")
                    continue

                img_count += 1
                save_path = os.path.join(out_person_dir, f"{person}_{img_count}.jpg")
                cv2.imwrite(save_path, face)
                print(f"Đã lưu khuôn mặt: {save_path}")

            except Exception as e:
                print(f"Lỗi xử lý {fname}: {e}")
                continue

    print(f"Đã cắt {img_count} khuôn mặt cho {person}")

print("Hoàn tất cắt khuôn mặt vào 'dataset/'!")