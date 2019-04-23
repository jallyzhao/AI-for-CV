import cv2
import numpy as np

def median_Blur(img, kernal, padding_way):
    height, width = img.shape
    h, w = kernal.shape
    circles = int((h - 1) / 2)
    target_img = np.zeros((height, width), dtype=np.float)
    # padding
    if padding_way == 'REPLICA':
        padding = np.pad(img, ((circles, circles), (circles, circles)), 'edge')

    elif padding_way == 'ZERO':
        padding = np.pad(img, ((circles, circles), (circles, circles)), 'constant')
    else:
        print('choose right padding way')
        return
    # convoloution卷积操作
    for i in range(height):
        for j in range(width):
            target_img[i, j] = np.median(padding[i:i + h, j:j + w] * kernal)
    target_img = target_img.clip(0, 255)  # 重置矩阵中小于0大于255的数值为0和255
    target_img = np.rint(target_img).astype('uint8')  #np.rint根据四射五入取整
    return target_img

'''if __name__=="_main_":
    #读取图像并转换为数组
    img = cv2.imread('D:/picture/test1.jpg', 0)
    img=np.array(img)
    # sobel算子,分别是水平方向,垂直方向和不分方向
    sobel_x = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]))
    sobel_y = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    sobel = np.array(([-1, -1, 0], [-1, 0, 1], [0, 1, 1]))
    # prewitt各个方向上的算子
    prewitt_x = np.array(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))
    prewitt_y = np.array(([-1, -1, -1], [0, 0, 0], [1, 1, 1]))
    prewitt = np.array(([-2, -1, 0], [-1, 0, 1], [0, 1, 2]))
    # 拉普拉斯算子
    laplacian = np.array(([0, -1, 0], [-1, 4, -1], [0, -1, 0]))
    laplacian_2 = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]))
    kernal_list=('sobel_x','sobel_y','sobel','prewitt_x','prewitt_y','prewitt','laplacian','laplacian')'''
img = cv2.imread('D:/picture/test1.jpg', 0)
median_kernal = np.ones((5, 5))
target_img=median_Blur(img, median_kernal, 'REPLICA')
print(target_img)
cv2.imshow('target_img',target_img)
cv2.waitKey()
cv2.destroyAllWindows()
