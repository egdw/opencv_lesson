import cv2
import numpy as np
import cv2 as cv

demo = cv.imread("demo.jpg")
rotateMatrix = cv.getRotationMatrix2D((100,100),20,1)
result = cv.warpAffine(demo,M=rotateMatrix,dsize=(demo.shape[0],demo.shape[1]))
cv.imshow("demo2",result)
cv.waitKey(0)