import cv2
import numpy as np 
import glob
import os
cv2.namedWindow("haha",cv2.WINDOW_NORMAL)
for path in glob.glob("img/*"):
    print(path)
    img = cv2.imread(path)
    # img = cv2.py rMeanShiftFiltering(img, 10, 10)
    h,w = img.shape[:2]
    medsiz = int(int(min(h,w)/70)/2)*2+1
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # his = cv2.equalizeHist(gray)
    median = cv2.medianBlur(gray,3)
    blurred = cv2.GaussianBlur(median, (3, 3), 0)
    thres = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,7,2)
    cv2.imshow("thres_ori",thres)
    # ret3,thres = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # thres = cv2.adaptiveThreshold(blurred,255,cv2.BORDER_REPLICATE, cv2.THRESH_BINARY,5,2)
    # ret,thres = cv2.threshold(blurred,90,255,cv2.THRESH_BINARY)
    # ret2,thres = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    canny = cv2.Canny(blurred, 0, 50)
    cv2.imshow("canny_ori",canny)
    # print(medsiz)
    # print(path)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT,(5,1))

    v_img = canny.copy()
    h_img = canny.copy()

    # kernel = np.ones((5,5),np.uint8)
    # kernel1 = np.ones((5,5),np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # kernel2 = np.ones((1,1),np.uint8)
    
    
    canny = cv2.dilate(canny,kernel,iterations = 1)
    # contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(canny,contours,-1, (255,255,255), -1)
    canny = cv2.erode(canny,kernel ,iterations = 1)

    thres = cv2.dilate(thres,kernel3,iterations = 1)
    cv2.imshow("thres_cos",thres)
    # contours, hierarchy = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(thres,contours,-1, (255,255,255), -1)
    thres = cv2.erode(thres,kernel3 ,iterations = 1)
    cv2.imshow("thres_er",thres)


    dilation = cv2.dilate(canny,kernel3,iterations = 1)
    cv2.imshow("dilation_thres",dilation)
    # dilation = cv2.dilate(canny,kernel1,iterations = 1)
    erosion = cv2.erode(dilation,kernel3,iterations = 1)


    dilation_v = cv2.dilate(v_img,kernel_v,iterations = 1)
    erosion_v = cv2.erode(dilation,kernel_v,iterations = 1)
    cv2.imshow("erosion_v",erosion_v)
    dilation_h = cv2.dilate(h_img,kernel_h,iterations = 1)
    erosion_h = cv2.erode(dilation,kernel_h,iterations = 1)
    cv2.imshow("erosion_h",erosion_h)

    # erosion = erosion_h and erosion_v 
    # erosion = cv2.bitwise_and(erosion_h,erosion_v)
    # erosion = cv2.erode(erosion,kernel2,iterations = 1)
    # dilation = cv2.dilate(erosion,kernel1,iterations = 1)
    
    # cv2.imshow("ero_ori",erosion)

    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # draw = cv2.drawContours(cv2.cvtColor(erosion,cv2.COLOR_GRAY2BGR),contours,-1, (0,255,0), 1)
    draw = cv2.cvtColor(erosion,cv2.COLOR_GRAY2BGR)
    s_cont = []
    # for i, c in enumerate(contours):
        
    #     area = cv2.contourArea(c)
    #     s_cont.append(area)
    for i, c in enumerate(contours):
        contours_poly = cv2.approxPolyDP(c, 3, True)
        boundRect = cv2.boundingRect(contours_poly)
        # print("dai",boundRect[2],boundRect[3])
        s = boundRect[2]*boundRect[3]
        area = cv2.contourArea(c)
        
        if area< h*w/600:
            continue
        print("dien tich", area)
        # if s< h*w/500:
        #     continue
        cv2.rectangle(img,(int(boundRect[0]), int(boundRect[1])),(int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])),(0,255,0),2)
        cv2.drawContours(draw,contours,i, (0,255,0), 1)
        cv2.imshow("img",img )
        cv2.imshow("draw",draw)
        # cv2.imshow("ero",erosion)
        # cv2.waitKey()
    # cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imshow("ero",erosion)
    cv2.imshow("dialete",dilation )
    cv2.imshow("draw",draw)
    cv2.imshow("canny",canny)
    cv2.imshow("gray",gray)
    # cv2.imshow("huhu",his)
    cv2.imshow("median",median )
    cv2.imshow("thres",thres )
    cv2.imshow("blurred",blurred)
    cv2.imshow("img",img )
    cv2.waitKey()
