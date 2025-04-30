import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# Cấu hình
DATASET_DIR = 'dataset'
EMB_FILE = 'embeddings.npy'
LABEL_FILE = 'labels.npy'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model FaceNet
try:
    model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
    if os.path.exists('facenet_finetuned.pth'):
        model.load_state_dict(torch.load('facenet_finetuned.pth', map_location=DEVICE))
        print("Loaded finetuned FaceNet model")
    else:
        print("Warning: Finetuned model not found. Using pretrained model.")
    print(f"Loaded FaceNet model on {DEVICE}")
except Exception as e:
    print(f"Error loading FaceNet model: {e}")
    exit()

# Tiền xử lý ảnh
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def preprocess_image(img):
    if img is None or img.size == 0:
        print("Ảnh đầu vào trống hoặc lỗi")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = data_transforms(img_rgb)
    tensor_mean = img_tensor.mean().item()
    tensor_std = img_tensor.std().item()
    if tensor_mean == 0 and tensor_std == 0:
        print("Tensor không hợp lệ (toàn 0)")
        return None
    if not torch.isfinite(img_tensor).all():
        print("Tensor chứa giá trị không hợp lệ (NaN hoặc Inf)")
        return None
    return img_tensor.unsqueeze(0).to(DEVICE)

# Trích xuất embeddings
def extract_embeddings(dataset_dir):
    embeddings = []
    labels = []

    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            print(f"Bỏ qua: {person_dir} không phải thư mục")
            continue

        print(f"Xử lý nhãn: {person_name}")
        for img_name in os.listdir(person_dir):
            if not img_name.endswith('.jpg'):
                continue
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)

            if img is None or img.size == 0:
                print(f"Lỗi đọc ảnh: {img_path}")
                continue

            if img.shape[0] < 50 or img.shape[1] < 50:
                print(f"Ảnh quá nhỏ: {img_path}")
                continue

            img_tensor = preprocess_image(img)
            if img_tensor is None:
                print(f"Lỗi tiền xử lý ảnh: {img_path}")
                continue

            with torch.no_grad():
                embedding = model(img_tensor).squeeze().cpu().numpy()
                if np.all(embedding == 0):
                    print(f"Embedding không hợp lệ (toàn 0): {img_path}")
                    continue
                if not np.isfinite(embedding).all():
                    print(f"Embedding chứa NaN/Inf: {img_path}")
                    continue
                if embedding.shape[0] != 512:
                    print(f"Embedding sai kích thước: {img_path}, shape: {embedding.shape}")
                    continue

            embeddings.append(embedding)
            labels.append(person_name)

    if not embeddings:
        raise ValueError("Không có embedding nào được tạo! Kiểm tra dữ liệu hoặc mô hình.")

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)

    # Thống kê số lượng embeddings
    label_counts = Counter(labels)
    print("Số lượng embeddings cho mỗi nhãn:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
        if count < 50:
            print(f"Cảnh báo: Số lượng embeddings cho {label} quá ít ({count}).")

    # Kiểm tra số nhãn
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print("Cảnh báo: Chỉ có 1 nhãn, không đủ để huấn luyện phân loại!")

    # Tính độ tương đồng giữa các nhãn
    print("\nPhân tích độ tương đồng (cosine similarity):")
    for i, label1 in enumerate(unique_labels):
        mask1 = labels == label1
        emb1 = embeddings[mask1]
        mean_emb1 = np.mean(emb1, axis=0)
        for label2 in unique_labels[i+1:]:
            mask2 = labels == label2
            emb2 = embeddings[mask2]
            mean_emb2 = np.mean(emb2, axis=0)
            similarity = cosine_similarity([mean_emb1], [mean_emb2])[0][0]
            print(f"{label1} vs {label2}: {similarity:.3f}")

    # Kiểm tra độ phân tán trong nhãn
    print("\nĐộ phân tán embeddings trong mỗi nhãn:")
    for label in unique_labels:
        mask = labels == label
        emb = embeddings[mask]
        if len(emb) > 1:
            intra_sim = cosine_similarity(emb).mean()
            print(f"{label}: Trung bình cosine similarity trong nhãn: {intra_sim:.3f}")

    return embeddings, labels

if __name__ == "__main__":
    print("Đang trích xuất embeddings...")
    try:
        embeddings, labels = extract_embeddings(DATASET_DIR)
        np.save(EMB_FILE, embeddings)
        np.save(LABEL_FILE, labels)
        print(f"Đã lưu {len(labels)} embeddings vào '{EMB_FILE}' và labels vào '{LABEL_FILE}'.")
    except Exception as e:
        print(f"Lỗi khi trích xuất embeddings: {e}")