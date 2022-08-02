import cv2
import numpy as np
import os


def average_filter(img):
    h, w, v = img.shape[:3]
    img2 = np.zeros((h, w, v), np.uint8)

    for i in range(1, h-1):
        for j in range(1, w-1):
            for channel in range(v):
                average = ((img[i-1, j-1, channel] / 9) + (img[i, j-1, channel] / 9) + (img[i+1, j-1, channel] / 9) + \
                           (img[i-1, j, channel] / 9) + (img[i, j, channel] / 9) + (img[i+1, j, channel] / 9) + \
                           (img[i-1, j+1, channel] / 9) + (img[i, j+1, channel] / 9) + (img[i+1, j+1, channel] / 9)) // 1
                img2[i, j, channel] = average
    img2 = img2.astype(np.uint8)
    return img2 

def median_filter(img):
    h, w, v = img.shape[:3]
    img3 = np.zeros((h, w, v), np.uint8)

    for i in range(1, h-1):
        for j in range(1, w-1):
            for channel in range(v):
                templist = [img[i-1, j-1, channel], img[i, j-1, channel], img[i+1, j-1, channel], img[i-1, j, channel], img[i, j, channel], img[i+1, j, channel], img[i-1, j+1, channel], img[i, j+1, channel], img[i+1, j+1, channel]]
                templist = sorted(templist)
                img3[i, j, channel] = templist[4]
    img3 = img3.astype(np.uint8)
    return img3

def gaussian_filter(img):
    h, w, v = img.shape[:3]
    img4 = np.zeros((h, w, v), np.uint8)
    kernel = [[1/16, 1/8, 1/16],
              [1/8, 1/4, 1/8],
              [1/16, 1/8, 1/16]]

    for i in range(1, h-1):
        for j in range(1, w-1):
            for channel in range(v):
                gaussian_value = img[i-1, j-1, channel]*kernel[0][0] + img[i, j-1, channel]*kernel[0][1] + img[i+1, j-1, channel]*kernel[0][2] + \
                                 img[i-1, j, channel]*kernel[1][0] + img[i, j, channel]*kernel[1][1] + img[i+1, j, channel]*kernel[1][2] + \
                                 img[i-1, j+1, channel]*kernel[2][0] + img[i, j+1, channel]*kernel[2][1] + img[i+1, j+1, channel]*kernel[2][2]
                img4[i, j, channel] = gaussian_value
    img4 = img4.astype(np.uint8)
    return img4

def Laplace_filter(img):
    h, w, v = img.shape[:3]
    img5 = np.zeros((h, w, v), np.uint8)
    kernel = [[-1, -1, -1],
              [-1, 8, -1],
              [-1, -1, -1]]

    for i in range(1, h-1):
        for j in range(1, w-1):
            for channel in range(v):
                gaussian_value = img[i-1, j-1, channel]*kernel[0][0] + img[i, j-1, channel]*kernel[0][1] + img[i+1, j-1, channel]*kernel[0][2] + \
                                 img[i-1, j, channel]*kernel[1][0] + img[i, j, channel]*kernel[1][1] + img[i+1, j, channel]*kernel[1][2] + \
                                 img[i-1, j+1, channel]*kernel[2][0] + img[i, j+1, channel]*kernel[2][1] + img[i+1, j+1, channel]*kernel[2][2]
                img5[i, j, channel] = gaussian_value
    img5 = img5.astype(np.uint8)
    return img5

def sobel_filter(img):
    h, w, v = img.shape[:3]
    img6 = np.zeros((h, w, v), np.float64)

    kernelx = [[1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]]
    
    kernely = [[1, 2, 1],
              [0, 0, 0],
              [-1, -2, -1]]

    for i in range(1, h-1):
        for j in range(1, w-1):
            for channel in range(v):
                sobel_x = img[i-1, j-1, channel]*kernelx[0][0] + img[i, j-1, channel]*kernelx[0][1] + img[i+1, j-1, channel]*kernelx[0][2] + \
                          img[i-1, j, channel]*kernelx[1][0] + img[i, j, channel]*kernelx[1][1] + img[i+1, j, channel]*kernelx[1][2] + \
                          img[i-1, j+1, channel]*kernelx[2][0] + img[i, j+1, channel]*kernelx[2][1] + img[i+1, j+1, channel]*kernelx[2][2]

                sobel_y = img[i-1, j-1, channel]*kernely[0][0] + img[i, j-1, channel]*kernely[0][1] + img[i+1, j-1, channel]*kernely[0][2] + \
                          img[i-1, j, channel]*kernely[1][0] + img[i, j, channel]*kernely[1][1] + img[i+1, j, channel]*kernely[1][2] + \
                          img[i-1, j+1, channel]*kernely[2][0] + img[i, j+1, channel]*kernely[2][1] + img[i+1, j+1, channel]*kernely[2][2]
                if (abs(sobel_x) + abs(sobel_y)) < 200:
                    img6[i, j, channel] = abs(sobel_x) + abs(sobel_y)
                else:
                    img6[i, j, channel] = 200
    img6 = img6.astype(np.uint8)
    return img6


def main():
    file_name = os.path.join(os.path.dirname(__file__), 'Lenna.jpg')
    assert os.path.exists(file_name)
    img = cv2.imread(file_name)
    img2 = average_filter(img)
    img3 = median_filter(img)
    img4 = gaussian_filter(img)
    img5 = Laplace_filter(img)
    img6 = sobel_filter(img)
    cv2.imshow("img_origin", img)
    cv2.imshow("img_average", img2)
    cv2.imshow("img_median", img3)
    cv2.imshow("img_guassian", img4)
    cv2.imshow("img_Laplace", img5)
    cv2.imshow("img_sobel", img6)
    cv2.waitKey(0)


main()
