import cv2 as cv
import numpy as np
img = cv.imread("demo.jpg")

img = cv.bitwise_not(img,img)
cv.imshow("demo",img)
cv.waitKey(0)

img = cv.bitwise_and(img,img)
cv.imshow("demo",img)
cv.waitKey(0)


img = cv.bitwise_or(img,img)
cv.imshow("demo",img)
cv.waitKey(0)


img = cv.bitwise_xor(img,img)
cv.imshow("demo",img)
cv.waitKey(0)