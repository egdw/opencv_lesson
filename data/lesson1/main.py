# Opencv初始的简单实用
import cv2 as cv

cv.namedWindow("demo",cv.WINDOW_AUTOSIZE)

img = cv.imread("demo.jpg")

cv.imshow("demo",img)

cv.waitKey(0)
