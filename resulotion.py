import cv2
import os
import numpy as np


for name in open('cfd.list'):
    name = name.strip()
    img = cv2.imread(name)
    resolution_size = (70,70)
    faceimg = cv2.resize(img, resolution_size)
    savename = './7070/' + name.split('/')[-1]
    cv2.imwrite(savename,faceimg)
