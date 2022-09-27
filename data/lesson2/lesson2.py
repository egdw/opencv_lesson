
import cv2 as cv
import numpy as np
# 扣取图像 截取图像

# img = cv.imread("./data/lesson2/demo.jpg")
# cv.namedWindow("demo", cv.WINDOW_AUTOSIZE)
# corpImg = img[100:200, 100:200, 1]
#
# # 图片hstack 和 vstack的使用。
# himg = np.hstack([img[100:200, 100:200, 0],img[100:200, 100:200, 1],img[100:200, 100:200, 2]])
# cv.imshow("demo", himg)
# cv.waitKey(0)

# 创建白色背景

white_background = np.full_like(np.ones((500,500,3)),255)
print(white_background.shape)
cv.putText(white_background,"aaa",(100,100),fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale=2,color=(0,0,0))
cv.circle(white_background,(110,110),20,color=(255,255,0),thickness=1)
cv.rectangle(white_background,(100,100),(200,200),color=(255,0,255),thickness=1)
cv.line(white_background, (100,100), (200, 200), color=(255, 0, 255), thickness=1)
cv.ellipse(white_background,(200,200),axes=(100,100),angle=20,startAngle=0,endAngle=120,color=(0,255,255))
triangles = np.array([
    [(150, 240), (95, 333), (205, 333)],
    [(60, 160), (20, 217), (100, 217)]])
rects = np.array([
    [(150,240),(200,200),(300,300)]
])


cv.fillPoly(white_background,rects,color=(255,255,0))

cv.namedWindow("demo", cv.WINDOW_AUTOSIZE)
cv.imshow("demo",white_background)
cv.waitKey(0)
