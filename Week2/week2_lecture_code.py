import cv2
import numpy as np

img=cv2.imread('D:/picture/lenna.jpg')
#img=cv2.imread('D:/picture/test1.jpg')
#img_cor=cv2.imread('D:/picture/corner1.png')

'''cv2.imshow('img',img)
#高斯模糊，比较优化与否时间差，不同大小卷积核和标准差的模糊效果
print(cv2.useOptimized())
e1=cv2.getTickCount() #计算运行时间
#Gaussian Kernal Effect
g_img1 = cv2.GaussianBlur(img,(7,7),5) # 5表示标准差，趋势的确是高斯矩阵尺寸和标准差越大，图像越模糊
cv2.imshow('gaussian_blur_lenna1',g_img1)
e2=cv2.getTickCount()
t=(e2-e1)/cv2.getTickFrequency() #return 时钟频率
print ('Optimization is True',t)

cv2.setUseOptimized(False)
e1=cv2.getTickCount() #计算运行时间
#Gaussian Kernal Effect
g_img1 = cv2.GaussianBlur(img,(7,7),5) # 5表示标准差，趋势的确是高斯矩阵尺寸和标准差越大，图像越模糊
cv2.imshow('gaussian_blur_lenna1',g_img1)
e2=cv2.getTickCount()
t=(e2-e1)/cv2.getTickFrequency() #return 时钟频率
print ('Optimization is False',t)

#图像变更模糊
g_img2 =cv2.GaussianBlur(img,(17,17),5)
cv2.imshow('gaussian_blur_lenna2',g_img2)

#图像更清晰，减小方差，图像尖锐，中心点起的
g_img3=cv2.GaussianBlur(img,(7,7),1)
cv2.imshow('gaussian_blur_lenna3',g_img3)
cv2.waitKey()
cv2.destroyAllWindows()'''
#查看高斯核

'''kernel=cv2.getGaussianKernel(7,1) #kernal是一维的，为了运算快
print(kernel)

#check if optimization is enabled
#print(cv2.useOptimized())
g1_img=cv2.GaussianBlur(img,(17,17),5)
g2_img=cv2.sepFilter2D(img,-1,kernel,kernel)
cv2.imshow('g1_blur_lenna',g1_img)
cv2.imshow('g2_blur_lenna',g2_img)
cv2.waitKey()
cv2.destroyAllWindows()
##########Other Application########
#2nd derivation:Laplacian
#kernel_lap=np.array([[0,1,0],[1,-4,1],[0,1,0]],np.float32)#亮度为0
#kernel_lap=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)#亮度增加
#kernel_lap=np.array([[1,1,1],[1,-8,1],[1,1,1]],np.float32)# sharper
#lap_img=cv2.filter2D(img,-1,kernel=kernel_lap)
##########Edge######
#x axis
edgex=np.array([[-1,2,-1],[0,0,0],[1,2,1]],np.float32)
edgey=np.array([[-1,0,-1],[-2,0,2],[-1,0,1]],np.float32)
sharpx_img=cv2.filter2D(img,-1,kernel=edgex)
sharpy_img=cv2.filter2D(img,-1,kernel=edgey)
cv2.imshow('edgex_lenna',sharpx_img)
cv2.imshow('edgey_lenna',sharpy_img)'''
##########角点###########
'''cv2.imshow('corner',img_cor)

img_cor_gray=np.float32(cv2.cvtColor(img_cor,cv2.COLOR_BGR2GRAY))
print('img_cor_gray\n',img_cor_gray)
cv2.imshow('img_cor_gray',img_cor_gray)
img_harris=cv2.cornerHarris(img_cor_gray,2,3,0.05)
cv2.imshow('img_harris',img_harris)
img_harris=cv2.dilate(img_harris,None)
thres = 0.05*np.max(img_harris)
img_cor[img_harris>thres]=[0,0,255]
cv2.imshow('img_harries',img_cor)'''
#############SIFT############
#creat sift class
sift=cv2.xfeatures2d.SIFT_create()
#detect SIFT
kp=sift.detect(img,None) #None for mask
#compute SIFT descriptor
kp,des=sift.compute(img,kp)
print(des.shape)
img_sift=cv2.drawKeypoints(img,kp,outImage=np.array([]),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.waitKey()
cv2.destroyAllWindows()
