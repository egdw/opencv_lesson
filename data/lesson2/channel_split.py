import cv2 as cv
import numpy as np

img = cv.imread("demo.jpg")

# 分割
print(img.shape)  # (294,500,3)
arr = np.array(cv.split(img)) # 分离
print(arr.shape)  # (3,294,500)

result = np.hstack([arr[0],arr[1],arr[2]])
cv.imshow("demo",result)
cv.waitKey(0)

# 合并
img = cv.merge(arr)
cv.imshow("demo",img)
cv.waitKey(0)
