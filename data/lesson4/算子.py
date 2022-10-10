import time

import cv2
import numpy as np

img = cv2.imread("img.png",cv2.IMREAD_GRAYSCALE)


def gaussian_noise(image, mean=0.1, sigma=0.1):
    """
    添加高斯噪声
    :param image:原图
    :param mean:均值
    :param sigma:标准差 值越大，噪声越多
    :return:噪声处理后的图片
    """
    image = np.asarray(image / 255, dtype=np.float32)  # 图片灰度标准化
    noise = np.random.normal(mean, sigma, image.shape).astype(dtype=np.float32)  # 产生高斯噪声
    output = image + noise  # 将噪声和图片叠加
    output = np.clip(output, 0, 1)
    output = np.uint8(output * 255)
    return output


def salt_noise(image):
    image = np.asarray(image / 255, dtype=np.float32)
    noise = np.random.randint(-1, 2, image.shape)
    print(noise)
    output = image + noise  # 将噪声和图片叠加
    print(image)
    output = np.clip(output, 0, 1)
    output = np.uint8(output * 255)
    return output


# guassian_img = gaussian_noise(img)
# 查找高斯滤波器系数
# blur = cv2.GaussianBlur(guassian_img, (5, 5), 0)

# sobel
sobelxy = cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)

#
img = cv2.imread('img.png',cv2.IMREAD_GRAYSCALE)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
# 参数1：src1，第一个原数组.
# 参数2：alpha，第一个数组元素权重
#
# 参数3：src2第二个原数组
# 参数4：beta，第二个数组元素权重
# 参数5：gamma，图1与图2作和后添加的数值。不要太大，不然图片一片白。总和等于255以上就是纯白色了。
sobelxy = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
#
laplacian = cv2.Laplacian(img,cv2.CV_64F)
#
# result = cv2.hconcat([sobelxy,sobelxy,laplacian])
# blur = cv2.GaussianBlur(guassian_img,(10,10),sigmaX=0)
cv2.imshow("guassian", laplacian)
cv2.waitKey(0)
