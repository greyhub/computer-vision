# 1. Đọc model đã lưu từ train.py
# 2. Đọc các ảnh test và extract descriptor
# 3. Xây dựng BOW với bộ test
# 4. Phân loại



import cv2
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from scipy.cluster.vq import vq
import joblib

# 1. Đọc mô hình đã lưu
# đọc mô hình đã chạy ở file train.py
clf, classes_names, stdSlr, k, voc = joblib.load("sift500_coil100.pkl")


# 2. Đọc các ảnh test và extract descriptor
# path tới folder test
test_path = 'coil-100-BOW/test'

image_paths = []
image_classes = []
class_id = 0

def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

for class_name in classes_names:
    dir = os.path.join(test_path, class_name)
    class_path = imglist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1
des_list = []
sift = cv2.SIFT_create(128)
for image_path in image_paths:
    im = cv2.imread(image_path)
    # im = cv2.resize(im, (150,150))
    kpts, des = sift.detectAndCompute(im, None)
    des_list.append((image_path, des))


# 3. Xây dựng BOW cho bộ test

test_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    if des_list[i][1] is not None:
        words, distance = vq(des_list[i][1], voc)
        for w in words:
            test_features[i][w] += 1
test_features = stdSlr.transform(test_features)

# 4. Phân loại
pred = clf.predict(test_features)
accuracy = accuracy_score(image_classes, pred)
print(accuracy)
