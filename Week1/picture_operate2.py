#平移，旋转，仿射变换
import cv2
import numpy as np
import random
img =cv2.imread('D:/picture/test1.jpg',0)
rows,cols=img.shape
#平移
M_zoom=np.float32([[1,0,100],[0,1,50]])
dst_zoom=cv2.warpAffine(img,M_zoom,(cols,rows))

#旋转
M_rotate=cv2.getRotationMatrix2D((cols/2,rows/2),90,0.5)
dst_rotate=cv2.warpAffine(img,M_rotate,(cols,rows))

#仿射变换  变换之后，平行线保持平行
img = cv2.imread('D:/picture/test1.jpg')
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M_affine = cv2.getAffineTransform(pts1,pts2)
dst_affine = cv2.warpAffine(img,M_affine,(cols,rows))

#change color
def random_light_color(img):
    #brightness
    B,G,R = cv2.split(img)
    b_rand = random.randint(-50,50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255-b_rand
        B[B>lim] = 255
        B[B<=lim] = (b_rand+B[B<=lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0- b_rand
        B[B<lim] = 0
        B[B>= lim] = (b_rand+B[B>=lim]).astype(img.dtype)

    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)

    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)
    img_merge=cv2.merge((B,G,R))
    return img_merge
img_randm_color=random_light_color( img)

##display window
cv2.namedWindow('img_zoom',cv2.WINDOW_FREERATIO)
cv2.imshow('img_zoom',dst_zoom)

cv2.namedWindow('img_rotate',cv2.WINDOW_FREERATIO)
cv2.imshow('img_rotate',dst_rotate)

cv2.imshow('img_affine',dst_affine)

cv2.imshow('img_random_color',img_randm_color)
cv2.waitKey()
cv2.destroyAllWindows()
