import cv2
import numpy as np

img = cv2.imread('D:/picture/test1.jpg', 0)
#cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.putText(img, " I an ZhaoZhijie", (300, 500), 0, 2, (0, 0, 255), 2)
cv2.imshow('image', img)
#cv2.resizeWindow('image',300,300)
k = cv2.waitKey()
if k == 27:
    cv2.destroyAllWindows()
    print('ESC')
else:
    cv2.destroyAllWindows()
    print(' NOTESC ')
# get picture attribute
print('img.shape', img.shape)
print('img.size =',img.size)
print(img.dtype)
img_zoom=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
img_crop = img[0:200,0:300]
cv2.imshow('img_crop',img_crop)
cv2.imshow('img_zoom',img_zoom)
cv2.waitKey()
cv2.destroyAllWindows()