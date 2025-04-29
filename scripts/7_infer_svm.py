import cv2
import os
import torch
import numpy as np
import joblib
import random
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from ultralytics import YOLO

# Config
yolo = YOLO('model_pretrained/yolov8s-face.pt')
yolo.conf = 0.15  # Giảm ngưỡng để tăng khả năng phát hiện khuôn mặt
clf = joblib.load('svm_model.pkl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
if os.path.exists('facenet_finetuned.pth'):
    facenet.load_state_dict(torch.load('facenet_finetuned.pth', map_location=device))
    print("Loaded finetuned FaceNet model")
else:
    print("Warning: Finetuned model not found. Using pretrained model.")

# Đồng bộ tiền xử lý
simple_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def prep_face(img):
    if img is None or img.size == 0:
        print("Ảnh đầu vào trống hoặc lỗi")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = simple_transform(img_rgb)
    tensor_mean = img_tensor.mean().item()
    tensor_std = img_tensor.std().item()
    if tensor_mean == 0 and tensor_std == 0:
        print("Tensor không hợp lệ (toàn 0)")
        return None
    if not torch.isfinite(img_tensor).all():
        print("Tensor chứa NaN/Inf")
        return None
    print(f"Tensor mean: {tensor_mean:.3f}, std: {tensor_std:.3f}")
    return img_tensor.unsqueeze(0).to(device)

# Map label -> color
label2color = {}

def get_color(label):
    if label not in label2color:
        color = tuple(random.randint(0, 255) for _ in range(3))
        label2color[label] = color
    return label2color[label]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Lỗi: Không thể mở webcam.")
    exit()

frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Lỗi: Không thể đọc frame từ webcam.")
        break
    frame = cv2.flip(frame, 1)

    detections = []
    results = yolo(frame)
    boxes = results[0].boxes
    coords = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    h, w, _ = frame.shape
    print(f"Frame {frame_counter}: Found {len(coords)} faces")

    for (x1, y1, x2, y2), cls in zip(coords, classes):
        if cls != 0:
            continue

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w-1, int(x2)), min(h-1, int(y2))

        if (x2 - x1) < 50 or (y2 - y1) < 50:
            print(f"Frame {frame_counter}: Face too small ({x2-x1}x{y2-y1})")
            continue

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            print(f"Frame {frame_counter}: Empty face region")
            continue

        img_tensor = prep_face(face)
        if img_tensor is None:
            print(f"Frame {frame_counter}: Failed to preprocess face")
            continue

        with torch.no_grad():
            emb = facenet(img_tensor).squeeze().detach().cpu().numpy().reshape(1, -1)
        if np.all(emb == 0) or not np.isfinite(emb).all():
            print(f"Frame {frame_counter}: Invalid embedding")
            continue

        prob = clf.predict_proba(emb).max()
        predicted_label = clf.predict(emb)[0]
        label = predicted_label if prob > 0.75 else 'Unknown'

        probs = clf.predict_proba(emb)[0]
        top_indices = np.argsort(probs)[-2:][::-1]
        top_labels = clf.classes_[top_indices]
        top_probs = probs[top_indices]
        print(f"Frame {frame_counter}: Probability: {prob:.3f}, Predicted: {predicted_label}, Final: {label}")
        print(f"Frame {frame_counter}: Top 2 probabilities: {top_labels[0]}: {top_probs[0]:.3f}, {top_labels[1]}: {top_probs[1]:.3f}")

        detections.append((x1, y1, x2, y2, label, prob))

    for x1, y1, x2, y2, label, prob in detections:
        color = get_color(label)
        text = f"{label} {prob*100:.1f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    if not detections:
        cv2.putText(frame, "No faces detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    frame_counter += 1
    cv2.imshow('Nhận diện khuôn mặt', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        print("Đang thoát...")
        break

cap.release()
cv2.destroyAllWindows()