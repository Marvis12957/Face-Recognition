import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from collections import Counter

# Load embeddings and labels
embeddings = np.load('embeddings.npy')
labels = np.load('labels.npy')

# Thống kê số lượng mẫu cho mỗi nhãn
print("Số lượng mẫu cho mỗi nhãn:")
print(Counter(labels))

# Thêm nhiễu nhỏ vào embeddings để tăng tổng quát hóa
noise_factor = 0.1
embeddings_noisy = embeddings + np.random.normal(0, noise_factor, embeddings.shape)
embeddings_noisy = np.clip(embeddings_noisy, -1, 1)

# Chia dữ liệu thành train (70%), validation (15%), và test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(embeddings_noisy, labels, test_size=0.15, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

# In số lượng mẫu trong mỗi tập
print(f"Số mẫu trong tập train: {len(X_train)}")
print(f"Số mẫu trong tập validation: {len(X_val)}")
print(f"Số mẫu trong tập test: {len(X_test)}")

# Định nghĩa tham số cho GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1.0],
    'gamma': ['scale', 'auto', 0.0001, 0.001],
}

# Khởi tạo SVM với class_weight='balanced'
clf = SVC(kernel='rbf', probability=True, class_weight='balanced')

# Tìm tham số tối ưu bằng GridSearchCV
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# In tham số tốt nhất
print("\nTham số tốt nhất từ GridSearchCV:")
print(grid_search.best_params_)
print(f"Độ chính xác tốt nhất (cross-validation): {grid_search.best_score_:.3f}")

# Lấy mô hình tốt nhất
best_clf = grid_search.best_estimator_

# Đánh giá trên tập train
train_pred = best_clf.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
print("\nĐánh giá trên tập train:")
print(f"Độ chính xác: {train_accuracy:.3f}")
print(classification_report(y_train, train_pred))
print("Ma trận nhầm lẫn (train):")
print(confusion_matrix(y_train, train_pred))

# Đánh giá trên tập validation
val_pred = best_clf.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)
print("\nĐánh giá trên tập validation:")
print(f"Độ chính xác: {val_accuracy:.3f}")
print(classification_report(y_val, val_pred))
print("Ma trận nhầm lẫn (validation):")
print(confusion_matrix(y_val, val_pred))

# Đánh giá trên tập test
test_pred = best_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)
print("\nĐánh giá trên tập test:")
print(f"Độ chính xác: {test_accuracy:.3f}")
print(classification_report(y_test, test_pred))
print("Ma trận nhầm lẫn (test):")
print(confusion_matrix(y_test, test_pred))

# Lưu mô hình tốt nhất
joblib.dump(best_clf, 'svm_model.pkl')
print("\nĐã lưu mô hình tốt nhất vào 'svm_model.pkl'")