import imghdr
import cv2
import numpy as np
import os

def erosion(img):
    h, w = img.shape[:2]
    SE = np.ones((3, 3), dtype=np.uint8)
    constant = 1
    imgErode= np.zeros((h, w), dtype=np.uint8)
    for i in range(constant, h-constant):
        for j in range(constant,w-constant):
            temp= img[i-constant:i+constant+1, j-constant:j+constant+1]
            p= temp*SE
            imgErode[i,j]= np.min(p)
    return imgErode


def dilation(img):
    h, w = img.shape[:2]
    imgDilate = np.zeros((h, w), dtype = np.uint8)
    SED = np.array([[0,1,0],[1,1,1],[0,1,0]])
    constant = 1
    
    for i in range(constant, h-constant):
        for j in range(constant, w-constant):
            temp = img[i-constant:i+constant+1, j-constant:j+constant+1]
            p = temp*SED
            imgDilate[i, j] = np.max(p)
    return imgDilate

def findEdge(img1, img2):
    return img1 - img2

def main():
    file_name = os.path.join(os.path.dirname(__file__), 'morphology_1.png')
    assert os.path.exists(file_name)
    img = cv2.imread(file_name, 0)
    img2 = dilation(img)
    img3 = erosion(img)
    img4 = erosion(img2)
    img5 = dilation(img3)
    img_findedge_photo1 = findEdge(img, img4)
    img_findedge_photo2 = findEdge(img5, img3)
    cv2.imshow('originThresh', img)
    cv2.imshow('dilation', img2)
    cv2.imshow('erosion', img3)
    cv2.imshow('close', img4)
    cv2.imshow('open', img5)
    cv2.imshow('edge1', img_findedge_photo1)
    cv2.imshow('edge2', img_findedge_photo2)
    cv2.waitKey(0)



main()