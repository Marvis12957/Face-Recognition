import cv2
import os
import re
import time
from ultralytics import YOLO

# Tạo thư mục nếu chưa có
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# Kiểm tra định dạng tên
def is_valid_name(name):
    return bool(re.match(r'^[a-zA-Z0-9]+$', name))

# Chụp ảnh tự động từ webcam
def capture_images_auto(name, base_folder='raw_images', max_images=50, delay=2):
    if not is_valid_name(name):
        print("Tên không hợp lệ! Chỉ dùng chữ cái và số, không dùng dấu cách hoặc ký tự đặc biệt.")
        return
    
    person_folder = os.path.join(base_folder, name)
    create_folder(person_folder)
    
    # Load YOLO để kiểm tra khuôn mặt
    yolo = YOLO('model_pretrained/yolov8s-face.pt')
    yolo.conf = 0.15  # Giảm ngưỡng để tăng khả năng phát hiện
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Lỗi: Không thể mở webcam.")
        return
    
    print(f"Đang chụp ảnh tự động cho {name}. Nhấn 'ESC' để dừng.")
    print(f"Thay đổi tư thế/góc độ/ánh sáng sau mỗi lần chụp ({delay} giây).")
    count = 0
    last_capture_time = time.time()
    
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
        cv2.imshow(f"Chụp ảnh tự động cho {name}", frame)
        
        # Chụp tự động sau mỗi khoảng thời gian delay
        current_time = time.time()
        if face_detected and (current_time - last_capture_time) >= delay:
            count += 1
            img_name = os.path.join(person_folder, f"{name}_{count}.jpg")
            cv2.imwrite(img_name, frame)
            print(f"Đã lưu ảnh: {img_name}")
            last_capture_time = current_time
        
        # Nhấn ESC để dừng
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    print(f"Tổng số ảnh đã lưu: {count}")
    cap.release()
    cv2.destroyAllWindows()
    print("Hoàn tất chụp ảnh.")

if __name__ == "__main__":
    name = input("Nhập tên của người (chỉ dùng chữ cái và số): ")
    capture_images_auto(name)