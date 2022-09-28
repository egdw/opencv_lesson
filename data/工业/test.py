import cv2 as cv

img = cv.imread("./data/real/img_3.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_blur = cv.GaussianBlur(gray, (3, 3), 0)


def change(val):
    pass


cv.namedWindow("demo", cv.WINDOW_AUTOSIZE)
cv.createTrackbar("threshold1", "demo", 0, 255, change)
cv.createTrackbar("threshold2", "demo", 0, 255, change)
cv.createTrackbar("apertureSize", "demo", 3, 7, change)

while True:
    threshold1 = cv.getTrackbarPos("threshold1", "demo")
    threshold2 = cv.getTrackbarPos("threshold2", "demo")
    apertureSize = cv.getTrackbarPos("apertureSize", "demo")
    bin = cv.Canny(img_blur, threshold1, threshold2, apertureSize=apertureSize)
    cv.imshow("demo", bin)
    q = cv.waitKey(10)
    if q == 27:
        break
