import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Tải dữ liệu từ file pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Lấy dữ liệu và nhãn
data = data_dict['data']
labels = data_dict['labels']

# Kiểm tra dữ liệu có phải là một danh sách các danh sách có kích thước đồng nhất không
# Nếu không, chúng ta sẽ xử lý bằng cách chuyển đổi các phần tử thành các mảng NumPy đồng nhất
# Cách xử lý đơn giản: Chuyển thành mảng NumPy nếu tất cả các phần tử có chiều dài đồng nhất

# Kiểm tra kích thước của từng phần tử trong data
max_len = max(len(d) for d in data)  # Lấy chiều dài của phần tử dài nhất

# Thêm padding (0) nếu cần thiết để dữ liệu có chiều dài đồng nhất
data = np.array([d + [0] * (max_len - len(d)) if len(d) < max_len else d for d in data])

# Chuyển labels thành mảng NumPy
labels = np.asarray(labels)

# Chia dữ liệu thành tập huấn luyện và kiểm tra
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Khởi tạo mô hình RandomForestClassifier
model = RandomForestClassifier()

# Huấn luyện mô hình
model.fit(x_train, y_train)

# Dự đoán kết quả
y_predict = model.predict(x_test)

# Tính toán độ chính xác
score = accuracy_score(y_predict, y_test)

# In ra kết quả
print('{}% of samples were classified correctly!'.format(score * 100))

# Lưu mô hình đã huấn luyện vào file pickle
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
