# 1. Đọc ảnh vào và tìm các descriptor
# 2. Phân cụm các descriptor này
# 3. Xây dựng tập BOW
# 4. Phân loại



import cv2
import numpy as np
import os
import time

# 1 Đọc. ảnh vào và tìm các descriptor
# path tới folder ảnh train
train_path = 'coil-100-BOW/train'
# list các classes
classes_name = os.listdir(train_path)

# list chứa đường dẫn tới các ảnh và class tương ứng
image_paths = []
image_classes = []
class_id = 0

def imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path)]

for class_name in classes_name:
    dir = os.path.join(train_path, class_name)
    class_path = imlist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

# list chứa các descriptor của ảnh
des_list = []
# sử dụng sift với 128 feature cho mỗi keypoint phát hiện trong ảnh
sift = cv2.SIFT_create(128)
# brisk = cv2.BRISK_create(30)


# Bước này đọc các ảnh và áp dụng sift lên ảnh
t1 = time.time()
for image_path in image_paths:
    im = cv2.imread(image_path)
    # im = cv2.resize(im, (150,150))
    kpts, des = sift.detectAndCompute(im, None)
    des_list.append((image_path, des))
t2 = time.time()
print("Done feature extraction in %d seconds" %(t2-t1))

# stack các descriptor lại để phân cụm
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    if descriptor is not None:
        descriptors = np.vstack((descriptors, descriptor))
descriptors_float = descriptors.astype(float)


# 2. Phân cụm các descriptor
# phân cụm các descriptor
from scipy.cluster.vq import kmeans, vq

# phân thành 500 cụm, giá trị voc trả về là 500 center của 500 cụm
k = 500
t3 = time.time()
voc, variance = kmeans(descriptors_float, k, 1)
t4 = time.time()
print("Done clustering in %d seconds" %(t4-t3))


# 3. Xây dựng tập BOW(tần suất xuất hiện của các cụm đã phân cụm ở trên ở từng ảnh)

im_features = np.zeros((len(image_paths), k), "float32")
# im_features[i][j]: số lượng cụm thứ j xuất hiện ở ảnh thứ i
for i in range(len(image_paths)):
    if des_list[i][1] is not None:
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            im_features[i][w] += 1

# chuẩn hóa histogram feature về mean = 0 và std = 1
from sklearn.preprocessing import StandardScaler
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# 4. Phân loại
# sử dụng SVM để phân train
# Tìm ra bộ tham số tốt nhất
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear']}

# clf = SVC(C = 100, gamma= 0.0001, kernel='rbf')
# clf.fit(im_features, np.array(image_classes))

# class_weight = {
#         0: (807 / (7 * 140)),
#         1: (807 / (7 * 140)),
#         2: (807 / (7 * 133)),
#         3: (807 / (7 * 70)),
#         4: (807 / (7 * 42)),
#         5: (807 / (7 * 140)),
#         6: (807 / (7 * 142))
#     }
t5 = time.time()
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3, scoring= 'accuracy', n_jobs=-1)
grid.fit(im_features, np.array(image_classes))
t6 = time.time()
print("Done classify in %d seconds" %(t6-t5))


# in ra bộ tham số tốt nhât
print(grid.best_params_)
# in ra độ chính xác tốt nhất
print(grid.best_score_)

# lưu lại mô hình
import joblib
joblib.dump((grid.best_estimator_, classes_name, stdSlr, k, voc), "sift500_coil100.pkl", compress=3)

