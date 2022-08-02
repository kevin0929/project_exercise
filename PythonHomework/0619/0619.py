import cv2
import numpy as np
import os

def img_sampling(img):
    h, w, v = img.shape[:3]
    img2 = np.zeros((w, h, v), dtype = np.uint8)
    for i in range(w):
        for j in range(h):
            for channel in range(v):
                if (i%2 == 0) and (j%2 == 0):
                    img2[i, j, channel] = img[i, j, channel]
    return img2

def img_down_sampling(img):
    h, w, v = img.shape[:3]
    factor = 2
    img3 = np.zeros((w // factor, h // factor, v), dtype = np.uint8)
    for i in range(0, w, factor):
        for j in range(0, h, factor):
            for channel in range(v):
                img3[i // factor, j // factor, channel] = img[i, j, channel]
    return img3

def img_up_sampling(img):
    h, w, v = img.shape[:3]
    factor = 2
    img4 = np.zeros((w * factor, h * factor, v), dtype=np.uint8)
    
    for i in range(0, w * factor, factor):            #先把邊界抓出來
        for j in range(0, h * factor, factor):
            for channel in range(v):
                img4[i, j, channel] = img[i // factor, j // factor, channel]   

    for i in range(0, w * factor, factor):      #把寬剩下的點給填滿
        for j in range(0, h * factor):
            for channel in range(v):
                img4[i:i+factor, j, channel] = img4[i, j, channel]

    for i in range(0, w * factor):              #把高剩下的點給填滿
        for j in range(0, h * factor, factor):
            for channel in range(v):
                img4[i, j:j+factor, channel] = img4[i, j, channel]
    
    return img4

def img_quantizing(img):
    h, w, v = img.shape[:3]
    img5 = np.zeros((w, h, v), np.uint8)
    num = [0, 32, 64, 96, 128, 160, 192, 224, 256]  #最大就256
    gray = 0
    for i in range(w):
        for j in range(h):
            for channel in range(v):        #256 / 8 = 32
                for k in range(len(num)):
                    if img[i][j][channel] < num[k]:
                        gray = num[k-1]
                        break

                img5[i][j][channel] = np.uint8(gray)
    return img5


def main():
    file_name = os.path.join(os.path.dirname(__file__), 'img.jpeg')
    assert os.path.exists(file_name)
    img = cv2.imread(file_name)
    img2 = img_sampling(img)
    img3 = img_down_sampling(img)
    img4 = img_up_sampling(img3)
    img5 = img_quantizing(img)
    cv2.imshow("img_origin", img)
    cv2.imshow("img_sampling", img2)
    cv2.imshow("img_down_sampling",img3)
    cv2.imshow("img_up_sampling", img4)
    cv2.imshow("img_quantizing", img5)
    cv2.waitKey(0)

main()