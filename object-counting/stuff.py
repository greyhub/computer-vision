import cv2
import numpy as np 
import glob
import time
# cv2.namedWindow("haha",cv2.WINDOW_NORMAL)
# cv2.namedWindow("concate",cv2.WINDOW_NORMAL)
for path in glob.glob("stuff/*"):
    t = time.time()
    print(path)
    img = cv2.imread(path)
    h,w = img.shape[:2]

    median = cv2.medianBlur(img,3)
    # blurred = cv2.GaussianBlur(median, (3, 3), 0)
    #tim canh
    canny = cv2.Canny(median, 20, 80)
    # cv2.imshow("canny_ori",canny)

    #open canh
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    canny = cv2.dilate(canny,kernel3,iterations = 1)
    canny = cv2.erode(canny,kernel3 ,iterations = 1)
    # cv2.imshow("canny",canny)
    #xu ly object o vien anh
    erosion = canny.copy()

    ero_all = cv2.flip(erosion,-1)
    ero_ver = cv2.flip(erosion,0)
    ero_hor = cv2.flip(erosion,1)

    new_1 = np.concatenate((ero_all,ero_ver,ero_all),axis=1)
    new_2 = np.concatenate((ero_hor,erosion,ero_hor),axis=1)
    new_3 = np.concatenate((ero_all,ero_ver,ero_all),axis=1)

    new_ero = np.concatenate((new_1,new_2,new_3),axis=0)

    # cv2.imshow("concate",new_ero)
    #tim contours
    contours_new, hierarchy = cv2.findContours(new_ero, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #fill contours
    cv2.drawContours(new_ero,contours_new,-1, (255,255,255), -1)

    # cv2.imshow("after",new_ero )
    h_new, w_new = new_ero.shape[:]
    # print("haha", h,w)

    x1 = int(w_new/3)
    x2 = int(2*w_new/3)
    y1 = int(h_new/3)
    y2 = int(2*h_new/3)

    final = new_ero[y1:y2,x1:x2]
    # cv2.imshow("okey",final )
    # print("haha", final.shape[:])

    #thuc hien phep Close Morphological de loai nhieu
    eros = cv2.erode(final,kernel2,iterations = 1)
    
    final = cv2.dilate(eros,kernel2,iterations = 1)
    # cv2.imshow("eros_thres",final)
    #tim contours
    contours, hierarchy = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        boundRect = cv2.boundingRect(contours_poly)
        area = cv2.contourArea(c)
        
        if area< h*w/900:
            continue
        count += 1
        cv2.rectangle(img,(int(boundRect[0]), int(boundRect[1])),(int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])),(0,255,0),2)
        cv2.drawContours(img,contours,i, (0,0,255), 1)


    # cv2.imshow("ero",erosion)
    # cv2.imshow("canny",canny)
    # cv2.imshow("gray",gray)
    # cv2.imshow("huhu",his)
    # cv2.imshow("median",median )
    # cv2.imshow("blurred",blurred)
    print("time: ", time.time()-t)
    cv2.imshow("img",img )
    print("so do dung: ",count)
    cv2.waitKey()
