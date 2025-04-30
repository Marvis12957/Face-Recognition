import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# Kiểm tra file đầu vào
if not os.path.exists('embeddings.npy') or not os.path.exists('labels.npy'):
    raise FileNotFoundError("Không tìm thấy file 'embeddings.npy' hoặc 'labels.npy'.")

# Load embeddings and labels
embeddings = np.load('embeddings.npy')
labels = np.load('labels.npy')

# Giảm chiều dữ liệu bằng t-SNE
perplexity = float(input("Nhập perplexity cho t-SNE (mặc định 20 hoặc tự động): ") or min(20, len(embeddings) // 5))
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
embeddings_2d = tsne.fit_transform(embeddings)

# Tạo scatter plot
plt.figure(figsize=(10, 8))
unique_labels = np.unique(labels)
colors = sns.color_palette("husl", len(unique_labels))

for i, label in enumerate(unique_labels):
    mask = labels == label
    count = np.sum(mask)
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                label=f"{label} ({count} samples)", color=colors[i], alpha=0.6, s=100)

plt.title("Scatter Plot of Face Embeddings (t-SNE)", fontsize=14)
plt.xlabel("t-SNE Component 1", fontsize=12)
plt.ylabel("t-SNE Component 2", fontsize=12)
plt.legend(title="Labels", fontsize=10)
plt.grid(True)
plt.tight_layout()

# Lưu biểu đồ
plt.savefig("embeddings_scatter_plot.png")
plt.show()

# Thống kê độ phân tán trong mỗi nhãn
print("\nĐộ phân tán của các điểm trong mỗi nhãn (khoảng cách trung bình đến tâm cụm):")
for label in unique_labels:
    mask = labels == label
    cluster = embeddings_2d[mask]
    if len(cluster) > 1:
        centroid = np.mean(cluster, axis=0)
        distances = np.sqrt(((cluster - centroid) ** 2).sum(axis=1))
        mean_distance = np.mean(distances)
        print(f"{label}: Mean distance to centroid = {mean_distance:.3f}")