import cv2 as cv
import numpy as np
img = cv.imread("demo.jpg")

ret,threshold = cv.threshold(img,150,maxval=255,type=cv.THRESH_BINARY)
print(ret)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret2,threshold2 = cv.threshold(gray,150,255,type=cv.THRESH_BINARY)
cv.imshow("demo",threshold2)
cv.waitKey(0)

threshold3 = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,5,0)
cv.imshow("demo",threshold3)
cv.waitKey(0)