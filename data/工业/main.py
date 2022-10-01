import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def change(val):
    pass

def return_rect_location(cnt):
    # 查找x，y的最大值和最小值
    x_min,y_min = (-1,1)
    x_max,y_max = (-1,1)
    # 左上，右上，左下，右下的顺序返回。


cv.namedWindow("demo", cv.WINDOW_AUTOSIZE)
img = cv.imread("./data/real/img.png")
img_width = img.shape[0]
img_height = img.shape[1]
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

# sobel 算法，提取矩阵轮廓
img_blur = cv.GaussianBlur(gray, (5, 5), 0)
bin = cv.Canny(img_blur, 0, 20, apertureSize=3)

# cv.imshow("demo", bin)
# cv.waitKey(0)

# 腐蚀
# dilate_img = cv.dilate(bin, (3, 3), iterations=3)
# erode_img = cv.erode(dilate_img, (3, 3), iterations=3)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
dilate_img = cv.morphologyEx(bin, cv.MORPH_CLOSE, kernel)
# cv.imshow("demo", dilate_img)
# cv.waitKey(0)

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
    # print(box_width, box_height)
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

    # 找小圆，确定当前的位置

    warped_gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(warped_gray, 30, 100, apertureSize=3)

    # 霍夫找圆

    # 找到小圆，定位当前图片位置
    circles = cv.HoughCircles(canny, method=cv.HOUGH_GRADIENT, dp=1, minDist=100
                              , param1=30, param2=20, minRadius=1, maxRadius=20)

    small_circle = None
    for circle in circles[0]:
        # cv.circle(warped, (int(circle[0]), int(circle[1])), radius=int(circle[2]), color=(255, 255, 0), thickness=2)
        small_circle = circle
        break
    # 以小圆作为标志物，对图片进行旋转，使得小圆总是出现在右下角
    warped_width = warped.shape[0]
    warped_height = warped.shape[1]
    if small_circle[0] < warped_width / 2 and small_circle[1] < warped_height / 2:
        # 当前小圆处于左上，旋转180
        warped = cv.rotate(warped, cv.ROTATE_180)
        canny = cv.rotate(canny, cv.ROTATE_180)
    elif small_circle[0] < warped_width / 2 and small_circle[1] > warped_height / 2:
        # 说明在左下需要进行水平flib
        warped = cv.flip(warped, 1)  # 水平翻转
        canny = cv.flip(canny, 1)  # 水平翻转

    elif small_circle[0] > warped_width / 2 and small_circle[1] < warped_height / 2:
        # 说明在右上，需要进行垂直翻转
        warped = cv.flip(warped, 0)
        canny = cv.flip(canny, 0)


    # 再找一次经过调整后的小圆位置
    circles = cv.HoughCircles(canny, method=cv.HOUGH_GRADIENT, dp=1, minDist=100
                              , param1=30, param2=20, minRadius=1, maxRadius=20)
    for circle in circles[0]:
        cv.circle(warped, (int(circle[0]), int(circle[1])), radius=int(circle[2]), color=(255, 255, 0), thickness=2)
        small_circle = circle
        break

    # 找到大圆
    circles = cv.HoughCircles(canny, method=cv.HOUGH_GRADIENT, dp=2, minDist=20
                              , param1=90, param2=90, minRadius=1, maxRadius=80)
    # print(circles)
    big_circle = None
    for circle in circles[0]:
        cv.circle(warped, (int(circle[0]), int(circle[1])), radius=int(circle[2]), color=(255, 255, 0), thickness=2)
        big_circle = circle
        break

    print("找到的大圆坐标", big_circle, " 找到的小圆坐标", small_circle)
    # 画出两个圆之间的距离,即a
    cv.line(warped,(int(big_circle[0]),int(big_circle[1])),(int(small_circle[0]),int(small_circle[1])),color=(255,0,255),thickness=2)


    # 计算夹角D
    # print(warped_width,warped_height)
    # 由于零件的大小是固定的，因此可以用上述的方法准确定位到左下角的位置
    # warped_left_bottem = warped[warped_width//2:warped_width,0:warped_height//15]
    # warped_left_bottem_canny = cv.Canny(warped_left_bottem, 30, 100, apertureSize=3)
    #
    # # cv.imshow("demo", warped_left_bottem_canny)
    # # cv.waitKey(0)
    #
    # lines = cv.HoughLines(warped_left_bottem_canny,rho=1,theta=np.pi/180,threshold=15)
    # print(lines)

    # print(warped_left_bottem.shape)
    #

    # 长得像钥匙。
    warped_key = warped[0:warped_width-warped_width//4, warped_height//2-warped_height//7:warped_height//2+warped_height//7]
    warped_key_canny = cv.Canny(warped_key, 30, 100, apertureSize=3)
    # 求上下高度b
    contoures, _ = cv.findContours(warped_key_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contoures:
        cnt_len = cv.arcLength(cnt, True)  # 计算轮廓周长
        cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)  # 多边形逼近
        print(cv.contourArea(cnt))
        if cv.contourArea(cnt) > 1000:
            cnt = cv.minAreaRect(cnt)
            data = np.array(cv.boxPoints(cnt), int)
            cnt = cv.boxPoints(cnt)
            box_height = int(np.sqrt(
                (cnt[1][0] - cnt[0][0]) ** 2 + (
                            cnt[1][1] - cnt[0][1]) ** 2))
            box_width = int(np.sqrt(
                (cnt[1][0] - cnt[2][0]) ** 2 + (
                            cnt[1][1] - cnt[2][1]) ** 2))
            # 得到居中的位置,这里还是要判断那边是右边。
            center_point = (int(cnt[1][0] + cnt[0][0])//2,int(cnt[1][1] + cnt[0][1])//2)
            # print(center_point)
            # cv.circle(warped, (warped_height//2-warped_height//7+center_point[0],center_point[1]), radius=10, color=(255, 255, 0), thickness=1)
            cv.line(warped,(warped_height//2-warped_height//7+center_point[0],0),(warped_height//2-warped_height//7+center_point[0],center_point[1]),color=(255,0,255),thickness=1)
            print("b:",center_point[1])
            break

    # 求长条圆柱
    warped_cycle = warped[warped_width-warped_width//4:warped_width-warped_width//10, warped_height//2-warped_height//7:warped_height//2+warped_height//3]
    warped_cycle_canny = cv.Canny(warped_cycle, 30, 100, apertureSize=3)
    # 求e
    contoures, _ = cv.findContours(warped_cycle_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contoures:
        cnt_len = cv.arcLength(cnt, True)  # 计算轮廓周长
        cnt = cv.approxPolyDP(cnt, 0.02 * cnt_len, True)  # 多边形逼近
        # print(cv.contourArea(cnt))
        if cv.contourArea(cnt) > 200:
            # print(cv.contourArea(cnt))
            cnt = cv.minAreaRect(cnt)
            data = np.array(cv.boxPoints(cnt), int)
            cnt = cv.boxPoints(cnt)
            box_height = int(np.sqrt(
                (cnt[1][0] - cnt[0][0]) ** 2 + (
                        cnt[1][1] - cnt[0][1]) ** 2))
            box_width = int(np.sqrt(
                (cnt[1][0] - cnt[2][0]) ** 2 + (
                        cnt[1][1] - cnt[2][1]) ** 2))

            # 得到左圆
            left_circle_center = (int(cnt[0][0]+box_width/2),int(box_width/2))
            # print("warped_cycle_canny:",left_circle_center)
            # cv.circle(warped_cycle_canny, left_circle_center, radius=int(box_width/2), color=(255, 255, 0), thickness=1)
            cv.circle(warped, (warped_height//2-warped_height//7+left_circle_center[0],warped_width-warped_width//4+left_circle_center[1]), radius=int(box_width/2), color=(255, 255, 0), thickness=1)

            # 得到右圆
            right_circle_center = (int(cnt[1][0] - box_width / 2), int(box_width / 2))
            # print("warped_cycle_canny:", right_circle_center)
            # cv.circle(warped_cycle_canny, right_circle_center, radius=int(box_width/2), color=(0, 255, 255), thickness=1)
            cv.circle(warped, (warped_height//2-warped_height//7+right_circle_center[0],warped_width-warped_width//4+right_circle_center[1]), radius=int(box_width/2), color=(255, 255, 0), thickness=1)


            cv.line(warped,(warped_height//2-warped_height//7+left_circle_center[0],warped_width-warped_width//4+left_circle_center[1]),
                    (warped_height//2-warped_height//7+right_circle_center[0],warped_width-warped_width//4+right_circle_center[1]),
                    color=(255, 0, 255),thickness=1)
            print("e:",box_height-box_width)
            # print(cnt[1][0] ,cnt[0][0])
            # # 得到居中的位置
            # center_point = (int(cnt[1][0] + cnt[0][0]) // 2, int(cnt[1][1] + cnt[0][1]) // 2)
            # print(box_height,box_width)
            # # cv.circle(warped, (warped_height//2-warped_height//7+center_point[0],center_point[1]), radius=10, color=(255, 255, 0), thickness=1)
            # cv.line(warped, (warped_height // 2 - warped_height // 7 + center_point[0], 0),
            #         (warped_height // 2 - warped_height // 7 + center_point[0], center_point[1]), color=(255, 0, 255),
            #         thickness=1)
            # print("b:", center_point[1])
            break

    cv.imshow("demo", warped)
    cv.waitKey(0)
