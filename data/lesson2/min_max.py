import cv2 as cv
import numpy as np

img = cv.imread("demo.jpg")
print(img.shape)
img2 = cv.imread("demo2.jpg")
# 对两个图片进行相同维度操作
img = cv.resize(img,(294,500))
img2 = cv.resize(img2,(294,500))
img3 = cv.min(img, img2)
result = np.hstack([img,img2,img3])
cv.imshow("demo",result)
cv.waitKey(0)

img3 = cv.max(img, img2)
result = np.hstack([img,img2,img3])
cv.imshow("demo",result)
cv.waitKey(0)