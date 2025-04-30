import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from PIL import Image
import numpy as np

# Định nghĩa dataset tùy chỉnh
class FaceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.labels = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if not self.labels:
            raise ValueError("Không tìm thấy thư mục nhãn nào trong dataset/.")
        
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.image_paths = []
        self.image_labels = []

        for label in self.labels:
            label_dir = os.path.join(data_dir, label)
            for img_name in os.listdir(label_dir):
                if img_name.endswith('.jpg'):
                    self.image_paths.append(os.path.join(label_dir, img_name))
                    self.image_labels.append(self.label_to_idx[label])

        if not self.image_paths:
            raise ValueError("Không tìm thấy ảnh nào trong dataset/.")

        print(f"Đã tìm thấy {len(self.labels)} nhãn: {self.labels}")
        print(f"Tổng số ảnh: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.image_labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Định nghĩa transform với augmentation
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
dataset = FaceDataset(data_dir='dataset', transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Load FaceNet pre-trained
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').to(device)
facenet.eval()

# Thêm lớp phân loại
class FaceNetClassifier(nn.Module):
    def __init__(self, facenet, num_classes):
        super(FaceNetClassifier, self).__init__()
        self.facenet = facenet
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        embedding = self.facenet(x)
        logits = self.fc(embedding)
        return logits, embedding

model = FaceNetClassifier(facenet, num_classes=len(dataset.labels)).to(device)

# Đóng băng 50% layer đầu tiên
total_layers = len(list(model.named_parameters()))
freeze_until = total_layers // 2
for idx, (name, param) in enumerate(model.named_parameters()):
    if idx < freeze_until and 'fc' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True

# Định nghĩa loss, optimizer, và scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

# Huấn luyện với early stopping
num_epochs = 20
patience = 5
best_val_accuracy = 0.0
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = train_correct / train_total

    # Đánh giá trên tập validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / val_total
    scheduler.step(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, '
          f'Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')

    # Early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(facenet.state_dict(), 'facenet_finetuned.pth')
        print(f"Saved best model with Val Accuracy: {best_val_accuracy:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

print("Finetuning completed!")