import numpy as np
import cv2 as cv
im = cv.imread('D:/Device-Control-Using-OpenCV-and-ML/Device-Control-Using-OpenCV-and-ML/opencv/book2.jpg')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#cv.imshow("kk",imgray)
#cv.waitKey(0)
#cv.destroyAllWindows()
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours,hierachy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(im, contours, -1, (0,255,0), 3)
cv.imshow("data is block",im)
cv.waitKey(0)
cv.destroyAllWindows()