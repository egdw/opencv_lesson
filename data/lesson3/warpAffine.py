import cv2
import numpy as np
import cv2 as cv

img = cv.imread("document.jpg")
# 四个坐标点
# 左上 （519，98）
# 右上 （1024，124）
# 左下 （302，810）
# 右下 （1119，832）
# 仿射变换
# 希望获得的图像的宽高
img_width = 300
img_height = 400

# 透视变换矩阵
perspectiveTransform = cv.getPerspectiveTransform(
    np.array([[519, 98],
              [1024, 124],
              [302, 810],
              [1119, 832]], dtype=np.float32)
    , np.array([[0, 0],
                [img.shape[0], 0],
                [0, img.shape[1]],
                [img.shape[0], img.shape[1]]], dtype=np.float32))

# 仿射变换矩阵
affineTransform = cv.getAffineTransform(np.array([[519, 98],
                                                  [1024, 124],
                                                  [302, 810]], np.float32),
                                        np.array([[0, 0],
                                                  [img_width, 0],
                                                  [0, img_height]], np.float32)
                                        )

# 图像仿射
warped = cv.warpAffine(img, M=affineTransform, dsize=(img_width, img_height))

warped2 = cv.warpPerspective(img,M=perspectiveTransform,dsize=(img.shape[0],img.shape[1]))

cv.imshow("warped", warped)
cv.waitKey(0)

cv.imshow("warped2",warped2)
cv.waitKey(0)

# 获取单应性矩阵
H = cv.findHomography(np.array([[519, 98],
              [1024, 124],
              [302, 810],
              [1119, 832]], dtype=np.float32),
                  np.array([[0, 0],
                            [img.shape[0], 0],
                            [0, img.shape[1]],
                            [img.shape[0], img.shape[1]]], dtype=np.float32)
                  ,method=cv.RANSAC)

if len(H)>1:
    H = H[0]

birdView = cv.warpPerspective(img,H,None,cv.INTER_LINEAR|cv.WARP_INVERSE_MAP|cv.WARP_FILL_OUTLIERS)

cv.imshow("birdView",birdView)
cv.waitKey(0)