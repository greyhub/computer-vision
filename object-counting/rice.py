import numpy as np
import cv2
import glob
# from matplotlib import pyplot as plt
import time
# cv2.namedWindow("concate",cv2.WINDOW_NORMAL)
# cv2.namedWindow("sgho1",cv2.WINDOW_NORMAL)
# cv2.namedWindow("thres_ori",cv2.WINDOW_NORMAL)
# cv2.namedWindow("haha",cv2.WINDOW_NORMAL)
# cv2.namedWindow("thres_cos",cv2.WINDOW_NORMAL)
# cv2.namedWindow("result_denoise",cv2.WINDOW_NORMAL)
cv2.namedWindow("count",cv2.WINDOW_NORMAL)
#ham khu nhieu o mien tan so
def denoisePeriodic(img, size_filter=2, diff_from_center_point=50,size_thresh=2):

    h,w = img.shape[:]
    img_float32 = np.fft.fft2(img)
    fshift = np.fft.fftshift(img_float32)

    #show the furier  image transform by log e of fft
    furier_tr = 20*np.log(np.abs(fshift))

    # plt.subplot(121),plt.imshow(img, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(furier_tr, cmap = 'gray')
    # plt.show()

    #get center point value
    center_fur = furier_tr[int(h/2)][int(w/2)]

    #find pick freq point
    new_fur = np.copy(furier_tr)
    kernel = np.ones((2*size_filter+1,2*size_filter+1),np.float32)/((2*size_filter+1)*(2*size_filter+1)-1)
    kernel[size_filter][size_filter]=-1
    kernel = -kernel
    # print(kernel)
    dst = cv2.filter2D(new_fur,-1,kernel)

    # plt.subplot(121),plt.imshow(dst, cmap = 'gray')
    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(dst, cmap = 'gray')
    # plt.subplot(122),plt.imshow(new_fur, cmap = 'gray')
    # plt.show()
    diff_from_center_point = center_fur*diff_from_center_point/356
    dst[0][:]=dst[1][:]=dst[:][0]=dst[:][1]=0

    dst[int(h/2)][int(w/2)]=0
    index = np.where(dst>diff_from_center_point)
    # print("index",index)

    # remove point isnot the pick one
    index_x = []
    index_y = []

    for i,item in enumerate(index[0]):
        
        value = furier_tr[index[0][i]][index[1][i]]
        # print("value ", value)
        matrix = np.copy(furier_tr[max(0,index[0][i]-size_filter):min(h,index[0][i]+size_filter+1),max(0,index[1][i]-size_filter):min(w,index[1][i]+size_filter+1)])
        # print("new maxtirx", matrix)
        matrix[size_filter][size_filter]=0
        
        max_value = np.amax(matrix)
        # print("mean", max_value)
        
        if (value-max_value<20):
            continue
        index_y.append(index[0][i])
        index_x.append(index[1][i])

        
    # print("to dau", index_x, index_y)
    # print("max freq", max_freq,center_fur)

    # set freq value of pick points to 1
    for i,item in enumerate(index_x):
        for j in range(size_thresh):
            for k in range(size_thresh):
                x = max(0,min(int(index_y[i]-int(size_thresh/2)+j),h-1))
                y = max(0,min(int(index_x[i]-int(size_thresh/2)+k),w-1))
                # print("toa do", x, y)
                furier_tr[x,y]=1
                fshift[x,y] = 1

    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(furier_tr, cmap = 'gray')

    # inverse to image
    # inverse shift
    f_ishift = np.fft.ifftshift(fshift)
    # inverse furier
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back).astype(np.uint8)
    # plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
    # plt.show()

    return img_back

#ham xu ly chung
def count_general(img):
    #khu nhieu chung
    median = cv2.medianBlur(img,5)
    blurred = cv2.GaussianBlur(median, (5, 5), 0)
    #mo rong anh de cac buoc loc khong anh huong cac vung o canh
    ero_all = cv2.flip(blurred,-1)
    ero_ver = cv2.flip(blurred,0)
    ero_hor = cv2.flip(blurred,1)

    new_1 = np.concatenate((ero_all,ero_ver,ero_all),axis=1)
    new_2 = np.concatenate((ero_hor,blurred,ero_hor),axis=1)
    new_3 = np.concatenate((ero_all,ero_ver,ero_all),axis=1)

    new_ero = np.concatenate((new_1,new_2,new_3),axis=0)
    # cv2.imshow("concate",new_ero) 

    h_new, w_new = new_ero.shape[:]
    # print("haha", h,w)
    x1 = int(w_new/3)
    x2 = int(2*w_new/3)
    y1 = int(h_new/3)
    y2 = int(2*h_new/3)  

    #can bang histogram cuc bo
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(int(w/50),int(h/50)))
    cl1 = clahe.apply(new_ero)
    cl1 = cv2.GaussianBlur(cl1, (5, 5), 0)
    # cv2.imshow("sgho1",cl1)

    #threshold cuc bo
    thres1 = cv2.adaptiveThreshold(cl1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,55,-12)
    # cv2.imshow("thres_ori",thres1)
    thres1 = thres1[y1:y2,x1:x2]
    # cv2.imshow("haha",thres1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    #co gian anh de loai nhieu va tach object lien nhau
    thres = cv2.erode(thres1,kernel ,iterations = 1)
    # cv2.imshow("thres_er",thres)
    thres = cv2.dilate(thres,kernel,iterations = 1)
    # cv2.imshow("thres_dia",thres)
    thres = cv2.erode(thres,kernel1,iterations = 1)
    # cv2.imshow("thres_cos",thres)

    #tim contours
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    #tim hat gao dien tich lon nhat
    max_area = 0
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        max_area = max(area,max_area)
    # print("max are", max_area)
    #ve, dem va loai nhung hat gao qua nho
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        boundRect = cv2.boundingRect(contours_poly)

        area = cv2.contourArea(c)
        if area< 0.08*max_area:
        # if area< 15:
            continue
        # print("dien tich", area)

        count += 1
        cv2.rectangle(img_ori,(int(boundRect[0]), int(boundRect[1])),(int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])),(0,255,0),1)
        cv2.drawContours(img_ori,contours,i, (0,0,255), 1)

    # cv2.imshow("img_ori",img_ori )
    # cv2.waitKey()
    return img_ori, count

#ham xu ly rieng cho anh khong can can bang sang va background co nhieu
def count_normal(img):
    #loc nhieu chung
    median = cv2.medianBlur(img,3)
    blurred = cv2.GaussianBlur(median, (5, 5), 0)
    # cv2.imshow("buller",blurred)
    # cv2.imshow("median",median)

    ero_all = cv2.flip(blurred,-1)
    ero_ver = cv2.flip(blurred,0)
    ero_hor = cv2.flip(blurred,1)

    new_1 = np.concatenate((ero_all,ero_ver,ero_all),axis=1)
    new_2 = np.concatenate((ero_hor,blurred,ero_hor),axis=1)
    new_3 = np.concatenate((ero_all,ero_ver,ero_all),axis=1)

    new_ero = np.concatenate((new_1,new_2,new_3),axis=0)
    #nhi phan anh cuc bo
    new_ero = cv2.adaptiveThreshold(new_ero,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,55,-5)

    h_new, w_new = new_ero.shape[:]
    # print("haha", h,w)

    x1 = int(w_new/3)
    x2 = int(2*w_new/3)
    y1 = int(h_new/3)
    y2 = int(2*h_new/3)

    thres1 = new_ero[y1:y2,x1:x2]
    # cv2.imshow("final", thres1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

    #co gian anh de loai nhieu va tach object lien nhau
    thres = cv2.erode(thres1,kernel ,iterations = 1)
    # cv2.imshow("thres_er",thres)
    thres = cv2.dilate(thres,kernel,iterations = 1)
    # cv2.imshow("thres_dia",thres)
    
    thres = cv2.erode(thres,kernel1,iterations = 1)
    # cv2.imshow("thres_cos",thres)

    #tim contours
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #dem so luong
    count = 0
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        boundRect = cv2.boundingRect(contours_poly)
        area = cv2.contourArea(c) 
        if area< 5:
            continue
        count += 1
        cv2.rectangle(img_ori,(int(boundRect[0]), int(boundRect[1])),(int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])),(0,255,0),1)
        cv2.drawContours(img_ori,contours,i, (0,0,255), 1)

    # cv2.imshow("img_ori",img_ori )
    # cv2.waitKey()
    return img_ori, count

#ham xu ly rieng cho anh phai can bang sang va background tuong doi it nhieu
def count_his(img):
    #loc nhieu chung
    median = cv2.medianBlur(img,3)
    blurred = cv2.GaussianBlur(median, (5, 5), 0)
    # cv2.imshow("buller",blurred)
    # cv2.imshow("median",median)
    #can bang histogram
    img = cv2.equalizeHist(blurred)
    # cv2.imshow("hist",img)

    #nhi phan anh
    thres = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,21,-2)
    # cv2.imshow("thres_ori",thres)
    #tim contours
    contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    max_area = 0
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        max_area = max(area,max_area)
    # print("max are", max_area)
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        boundRect = cv2.boundingRect(contours_poly)

        area = cv2.contourArea(c)       
        if area< 0.08*max_area:
            continue
        # print("dien tich", area)

        count += 1
        cv2.rectangle(img_ori,(int(boundRect[0]), int(boundRect[1])),(int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])),(0,255,0),1)
        cv2.drawContours(img_ori,contours,i, (0,0,255), 1)

    # cv2.imshow("img_ori",img_ori )
    # cv2.waitKey()
    return img_ori, count

for path in glob.glob("rice/*"):
    # img_ori = cv2.imread('good_1.png')
    print("image file: ", path)
    t = time.time()
    img_ori = cv2.imread(path)
    # img_ori = cv2.imread('histogram.png')
    # img_ori = cv2.imread('rice.png')
    # img_ori = cv2.imread('nice.png')
    img_ori2 = img_ori.copy()
    img_gray = cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY)
    h,w = img_gray.shape[:]
    thres = int(max(h,w)/160)
    # print(thres)
    img = denoisePeriodic(img_gray, size_filter=int(thres/2), diff_from_center_point=50,size_thresh=thres)
    img_denoise = img.copy()
    # cv2.imshow("result_denoise",img)
    # cv2.waitKey()
    res_img, res_count = count_general(img)
    print("time: ", time.time()-t)
    print("so hat gao",res_count)
    cv2.imshow("count",res_img)
    cv2.waitKey()





