# -*- encoding: utf-8 -*-
import cv2
import numpy as np

def main():
    global img
    img = np.zeros((300, 400, 3), np.uint8) #创建初始画布大小
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE) #创建窗口
    img[:] = [255, 255, 255] #初始化画布颜色
    # 创建一个开关滑动条，只有两个值，起开关按钮作用
    switch = '0:OFF\n1:ON'
    cv2.createTrackbar(switch, 'image', 0, 1, callback)
    #创建3个滑动条来返回r，g，b的值
    cv2.createTrackbar('R', 'image', 0, 255, callback)
    cv2.createTrackbar('B', 'image', 0, 255, callback)
    cv2.createTrackbar('G', 'image', 0, 255, callback)
    while (True):
        global r ,g,b
        r = cv2.getTrackbarPos('R', 'image') #得到滑动条返回值，用名字即可获取某个滑动条
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        if cv2.getTrackbarPos(switch, 'image') == 1:  #为1则画
            img[:] = [b,r,g]
        else:      #为0则清除为白色
            img[:] = [255, 255, 255]
        if cv2.waitKey(10) & 0xFF == 27:  #按esc退出
            break
        cv2.imshow('image', img)
    cv2.destroyAllWindows()

# 定义回调函数，此程序无需回调，所以Pass即可
def callback(object):  #注意这里createTrackbar会向其传入参数即滑动条地址（几乎用不到），所以必须写一个参数
    pass



if __name__ =="__main__":
    main()
