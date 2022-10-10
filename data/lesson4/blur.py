import time

import cv2
import numpy as np

img = cv2.imread("img.png")


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


guassian_img = gaussian_noise(img)
blur = cv2.blur(guassian_img,ksize=(3,3))
cv2.imshow("guassian", blur)
cv2.waitKey(0)
