import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("/Users/test/PycharmProjects/opencv_lesson/data/工业/test/demo.png")
img = img[:img.shape[0]//2,:img.shape[1]//3]
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
bin = cv.Canny(gray, 0, 100, apertureSize=3)
cv.imshow("demo",bin)
cv.waitKey(0)
lines = cv.HoughLinesP(bin,1,np.pi/180,1,lines=1,minLineLength=40,maxLineGap=200)
if lines is not None:
    for x1, y1, x2, y2 in lines[0]:
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imshow("demo",img)
    cv.waitKey(0)
