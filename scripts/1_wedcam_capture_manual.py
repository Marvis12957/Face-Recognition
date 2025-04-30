import cv2
import os
import re
import numpy as np
from ultralytics import YOLO

# Tạo thư mục nếu chưa có
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# Kiểm tra định dạng tên
def is_valid_name(name):
    return bool(re.match(r'^[a-zA-Z0-9 ]+$', name))

# Kiểm tra chất lượng ảnh
def is_good_quality(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return 50 < brightness < 200 and sharpness > 100

# Chọn webcam động
def open_webcam():
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            return cap, i
        cap.release()
    return None, -1

# Lấy ảnh từ webcam
def capture_images(name, base_folder='raw_images', max_images=50):
    if not is_valid_name(name):
        print("Tên không hợp lệ! Chỉ dùng chữ cái, số và dấu cách không dùng ký tự đặc biệt")
        return
    
    person_folder = os.path.join(base_folder, name)
    create_folder(person_folder)
    
    # Load YOLO để kiểm tra khuôn mặt
    yolo = YOLO('model_pretrained/yolov8s-face.pt')
    yolo.conf = 0.15
    
    cap, index = open_webcam()
    if cap is None:
        print("Lỗi: Không tìm thấy webcam.")
        return
    print(f"Sử dụng webcam index: {index}")
    
    print(f"Đang chụp ảnh cho {name}. Nhấn 'Space' để chụp, 'ESC' để dừng.")
    count = 0
    
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi: Không thể đọc frame từ webcam.")
            break
        
        frame = cv2.flip(frame, 1)
        
        # Kiểm tra khuôn mặt
        results = yolo(frame)
        face_detected = len(results[0].boxes.xyxy) > 0
        
        # Hiển thị webcam với số ảnh đã chụp
        status = "Face detected" if face_detected else "No face detected"
        cv2.putText(frame, f"Images: {count}/{max_images} | {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if face_detected else (0, 0, 255), 2)
        cv2.imshow(f"Chụp ảnh cho {name}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if face_detected:
                if is_good_quality(frame):
                    count += 1
                    img_name = os.path.join(person_folder, f"{name}_{count}.jpg")
                    cv2.imwrite(img_name, frame)
                    print(f"Đã lưu ảnh: {img_name}")
                else:
                    print("Ảnh chất lượng thấp, thử lại.")
            else:
                print("Không phát hiện khuôn mặt, thử lại.")
        
        if key == 27:
            break
    
    print(f"Tổng số ảnh đã lưu: {count}")
    if count < 50:
        print(f"Cảnh báo: Chỉ lưu được {count} ảnh. Nên thu thập thêm để đạt tối thiểu 50 ảnh.")
    cap.release()
    cv2.destroyAllWindows()
    print("Hoàn tất chụp ảnh.")

if __name__ == "__main__":
    name = input("Nhập tên của người (chỉ dùng chữ cái, số và đấu cách): ")
    capture_images(name)