import cv2
import numpy as np 
import glob
import os
cv2.namedWindow("haha",cv2.WINDOW_NORMAL)
for path in glob.glob("img/*"):
    print(path)
    img = cv2.imread(path)
    h,w = img.shape[:2]
    medsiz = int(int(min(h,w)/70)/2)*2+1
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray,5)
    blurred = cv2.GaussianBlur(median, (3, 3), 0)
    thres = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,7,2)
    cv2.imshow("thres_ori",thres)
    canny = cv2.Canny(blurred, 0, 300)
    cv2.imshow("canny_ori",canny)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT,(5,1))

    v_img = canny.copy()
    h_img = canny.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    
    canny = cv2.dilate(canny,kernel,iterations = 1)
    canny = cv2.erode(canny,kernel ,iterations = 1)

    thres = cv2.dilate(thres,kernel3,iterations = 1)
    cv2.imshow("thres_cos",thres)
    thres = cv2.erode(thres,kernel3 ,iterations = 1)
    cv2.imshow("thres_er",thres)


    dilation = cv2.dilate(canny,kernel3,iterations = 1)
    cv2.imshow("dilation_thres",dilation)
    erosion = cv2.erode(dilation,kernel3,iterations = 1)


    dilation_v = cv2.dilate(v_img,kernel_v,iterations = 1)
    erosion_v = cv2.erode(dilation,kernel_v,iterations = 1)
    cv2.imshow("erosion_v",erosion_v)
    dilation_h = cv2.dilate(h_img,kernel_h,iterations = 1)
    erosion_h = cv2.erode(dilation,kernel_h,iterations = 1)
    cv2.imshow("erosion_h",erosion_h)

    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw = cv2.cvtColor(erosion,cv2.COLOR_GRAY2BGR)
    s_cont = []
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        boundRect = cv2.boundingRect(contours_poly)
        s = boundRect[2]*boundRect[3]
        area = cv2.contourArea(c)
        
        if area< h*w/600:
            continue
        print("dien tich", area)
        cv2.rectangle(img,(int(boundRect[0]), int(boundRect[1])),(int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])),(0,255,0),2)
        cv2.drawContours(draw,contours,i, (0,255,0), 1)
        cv2.imshow("img",img )
        cv2.imshow("draw",draw)
    cv2.imshow("ero",erosion)
    cv2.imshow("dialete",dilation )
    cv2.imshow("draw",draw)
    cv2.imshow("canny",canny)
    cv2.imshow("gray",gray)
    cv2.imshow("median",median )
    cv2.imshow("thres",thres )
    cv2.imshow("blurred",blurred)
    cv2.imshow("img",img )
    cv2.waitKey()
