import cv2
import numpy as np
import cv2 as cv

demo = cv.imread("demo.jpg")
d1 = cv.flip(demo,0)
d2 = cv.flip(demo,1)
result = np.hstack([d1,d2])

cv2.imshow("demo",result)
cv2.waitKey(0)

# 拼接，传入的是数据
result2 = cv2.hconcat([d1,d2])
result3 = cv2.vconcat([d1,d2])
cv2.imshow("result2",result2)
cv2.waitKey(0)
cv2.imshow("result3",result3)
cv2.waitKey(0)