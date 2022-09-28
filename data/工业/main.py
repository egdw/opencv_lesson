import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("./data/real/img.png")
img_width = img.shape[0]
img_height = img.shape[1]
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# sobel 算法，提取矩阵轮廓
img_blur = cv.GaussianBlur(gray, (5, 5), 0)
bin = cv.Canny(img_blur, 0, 20, apertureSize=3)

cv.imshow("demo", bin)
cv.waitKey(0)

# 腐蚀
# dilate_img = cv.dilate(bin, (3, 3), iterations=3)
# erode_img = cv.erode(dilate_img, (3, 3), iterations=3)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
dilate_img = cv.morphologyEx(bin, cv.MORPH_CLOSE, kernel)
cv.imshow("demo", dilate_img)
cv.waitKey(0)

# 找到外部的矩形
contoures, _ = cv.findContours(dilate_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
rect_locations = None
for cnt in contoures:
    cnt_len = cv.arcLength(cnt, True)  # 计算轮廓周长
    cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)  # 多边形逼近
    # if cv.contourArea(cnt) > 1000 and cv.isContourConvex(cnt):
    if cv.contourArea(cnt) > 40000:
        print(cv.contourArea(cnt))
        cnt = cv.minAreaRect(cnt)
        data = np.array(cv.boxPoints(cnt), int)
        cnt = cv.boxPoints(cnt)
        rect_locations = cnt
        # cv.circle(img, data[0], radius=20, color=(255, 255, 0), thickness=10)
        # cv.circle(img, data[1], radius=20, color=(255, 255, 0), thickness=10)
        # cv.circle(img, data[2], radius=20, color=(255, 255, 0), thickness=10)
        # cv.circle(img, data[3], radius=20, color=(255, 255, 0), thickness=10)
        break

# 透视变换
# 判断图片的长宽比，对齐进行自动调整。
if rect_locations is None:
    print("查找失败")
else:
    box_width = int(np.sqrt(
        (rect_locations[1][0] - rect_locations[2][0]) ** 2 + (rect_locations[1][1] - rect_locations[2][1]) ** 2))
    box_height = int(np.sqrt(
        (rect_locations[1][0] - rect_locations[0][0]) ** 2 + (rect_locations[1][1] - rect_locations[0][1]) ** 2))
    print(box_width, box_height)
    perspectiveTransform = cv.getPerspectiveTransform(
        np.float32([rect_locations[1], rect_locations[2], rect_locations[0], rect_locations[3]]).squeeze()
        , np.float32([[0, 0],
                      [box_width, 0],
                      [0, box_height],
                      [box_width, box_height]]))
    warped = cv.warpPerspective(img, perspectiveTransform, (box_width, box_height))
    if box_width < box_height:  # 说明是竖着的
        # 进行转置。
        warped = cv.rotate(warped, cv.ROTATE_90_COUNTERCLOCKWISE)

    # 大圆

    cv.imshow("demo", warped)
    cv.waitKey(0)
