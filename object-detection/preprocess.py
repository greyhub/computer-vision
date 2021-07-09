# file này xử lý bộ coil100 sau khi tải về và chia thành train, test
# trước khi chạy file này thì dùng command line để tạo cây thư mục dạng chuẩn
# coil-100-BOW
#     train
#         obj1
#         obj2
#         obj3
#         ...
#         obj100
#     test
#         obj1
#         obj2
#         obj3
#         ...
#         obj100

import os
import cv2

# đọc folder gốc
file_names = os.listdir('coil-100-original')
# class_names = ['obj1', 'obj2', 'obj3', 'obj4', 'obj5', 'obj6', 'obj7', 'obj8', 'obj9', 'obj10',
# 'obj11', 'obj12', 'obj13', 'obj14', 'obj15', 'obj16', 'obj17', 'obj18', 'obj19', 'obj20',
# 'obj21', 'obj22', 'obj23', 'obj24', 'obj25', 'obj26', 'obj27', 'obj28', 'obj29', 'obj30',
# 'obj31', 'obj32', 'obj33', 'obj34', 'obj35', 'obj36', 'obj37', 'obj38', 'obj39', 'obj40',
# 'obj41', 'obj42', 'obj43', 'obj44', 'obj45', 'obj46', 'obj47', 'obj48', 'obj49', 'obj50',
# 'obj51', 'obj52', 'obj53', 'obj54', 'obj55', 'obj56', 'obj57', 'obj58', 'obj59', 'obj60',
# 'obj61', 'obj62', 'obj63', 'obj64', 'obj65', 'obj66', 'obj67', 'obj68', 'obj69', 'obj70',
# 'obj71', 'obj72', 'obj73', 'obj74', 'obj75', 'obj76', 'obj77', 'obj78', 'obj79', 'obj80',
# 'obj81', 'obj82', 'obj83', 'obj84', 'obj85', 'obj86', 'obj87', 'obj88', 'obj89', 'obj90',
# 'obj91', 'obj92', 'obj93', 'obj94', 'obj95', 'obj96', 'obj97', 'obj98', 'obj99', 'obj100']



train_test_count = 0
def get_class_name(file_name):
    pos = file_name.find('_')
    res = file_name[:pos]
    return res

# phân chia vào folder train/test tỉ lệ 80/20
for file_name in file_names:
    img = cv2.imread(os.path.join('coil-100-original', file_name))
    class_name = get_class_name(file_name)
    if train_test_count < 58:
        cv2.imwrite(os.path.join('coil-100-BOW/train', class_name, file_name), img)
        train_test_count += 1
    else:
        cv2.imwrite(os.path.join('coil-100-BOW/test', class_name, file_name), img)
        train_test_count += 1
        if train_test_count == 72:
            train_test_count = 0
