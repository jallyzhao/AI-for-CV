# gamma conversion 展示一张图，展示图像直方图，做gamma变换，再展示直方图
import cv2
import numpy as np
from matplotlib import pyplot as plt

img_dark = cv2.imread('D:/picture/test1.jpg',0)


def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255) ** invGamma) * 255)
    table = np.array(table).astype('uint8')
    return cv2.LUT(img_dark, table)


img_brighter = adjust_gamma(img_dark, 2)

cv2.imshow('img_dark', img_dark)
cv2.imshow('img_brighter', img_brighter)
cv2.waitKey()
cv2.destroyAllWindows()

# 直方图
img_small_brighter = cv2.resize(img_brighter, (int(img_brighter.shape[0] * 0.5), int(img_brighter.shape[1] * 0.5)))
img_small_dark = cv2.resize(img_dark, (int(img_dark.shape[0] * 0.5), int(img_dark.shape[1] * 0.5)))
#hist=cv2.calcHist([img_small_brighter],[0],None,[256],[0,256])
#plt.plot(hist)
#plt.show()
cv2.imshow('img_small_brighter', img_small_brighter)
cv2.imshow('img_small_dark', img_small_brighter)
cv2.waitKey()
cv2.destroyAllWindows()
plt.subplot(2,1,1)
plt.hist(img_small_brighter.flatten(), 256, [0, 256], color='r')  # flatten将二维数组转换成一维数组

plt.title('brighter_histogram')
plt.show()
plt.subplot(2,1,2)
dark_hist=plt.hist(img_small_dark.flatten(), 256, [0, 256], color='r')

plt.title('dark_histogram')
plt.show()



#cv2.waitKey()
#cv2.destroyAllWindows()
