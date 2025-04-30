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
print(f"Tổng số mẫu: {len(embeddings)}")
print(f"Số lượng lớp: {len(np.unique(labels))}")

# Chia dữ liệu thành train (70%), validation (15%), và test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(embeddings, labels, test_size=0.15, random_state=42, stratify=labels)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp)

# Hàm thêm nhiễu để tăng tổng quát hóa
def add_noise(X, factor):
    return np.clip(X + np.random.normal(0, factor, X.shape), -1, 1)

# Thêm nhiễu vào train và validation, KHÔNG thêm nhiễu vào test
X_train = add_noise(X_train, 0.3)
X_val = add_noise(X_val, 0.1)

# In số lượng mẫu trong mỗi tập
print(f"Số mẫu trong tập train: {len(X_train)}")
print(f"Số mẫu trong tập validation: {len(X_val)}")
print(f"Số mẫu trong tập test: {len(X_test)}")

# Định nghĩa tham số cho GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 1.0, 10, 100],
    'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1]
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

# Đánh giá trên các tập
for name, X, y in [('train', X_train, y_train), ('validation', X_val, y_val), ('test', X_test, y_test)]:
    y_pred = best_clf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"\nĐánh giá trên tập {name}:")
    print(f"Độ chính xác: {acc:.3f}")
    print(classification_report(y, y_pred))
    print(f"Ma trận nhầm lẫn ({name}):")
    print(confusion_matrix(y, y_pred))

# Lưu mô hình tốt nhất
joblib.dump(best_clf, 'svm_model.pkl')
print("\nĐã lưu mô hình tốt nhất vào 'svm_model.pkl'")