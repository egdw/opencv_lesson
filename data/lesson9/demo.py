import cv2
import numpy as np

# 图像灰度化
trainImg = cv2.imread('img.png')[:4000, :, :]
trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)
queryImg = cv2.imread('img_2.png')[:4000, :, :]
queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)


# 计算图像特征点
def detectAndDescribe(image):
    descriptor = cv2.ORB_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    return (kps, features)


kpsA, featuresA = detectAndDescribe(trainImg_gray)
kpsB, featuresB = detectAndDescribe(queryImg_gray)


def createMatcher():
    # crossCheck表示两个特征点相互匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return bf


def matchKeyPointsBF(featuresA, featuresB):
    bf = createMatcher()
    # 匹配特征点
    best_matches = bf.match(featuresA, featuresB)
    # 排序，按距离从小到大排序
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches

matches = matchKeyPointsBF(featuresA, featuresB)
# 前100个特征画出来
# 不画出单独的点。
img3 = cv2.drawMatches(trainImg,kpsA,queryImg,kpsB,matches[:100],
                       None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 计算旋转变换矩阵
def getHomography(kpsA, kpsB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

        # estimate the homography between the sets of points
        # ransacReprojThreshold 重投影最大误差
        (H, _) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return H
    else:
        return None

H = getHomography(kpsA, kpsB, matches, reprojThresh=4)

# 进行全景图透视变换
width = trainImg.shape[1] + queryImg.shape[1]
height = trainImg.shape[0] + queryImg.shape[0]

result = cv2.warpPerspective(trainImg, H, (width, height))
result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg

# 自动查找图片位置
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

#根据二值化对图像位置进行自动查找
cnts,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in cnts:
    area = cv2.contourArea(cnt)
    if area>100000:
        (x, y, w, h) = cv2.boundingRect(cnt)
        print(area,x,y,w,h)
        result = result[y:y + h, x:x + w]
        cv2.imshow("result",result)
        cv2.waitKey(0)
        break