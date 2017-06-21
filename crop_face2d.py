#!/usr/bin/env python
#
# crop face using yolo-Sunery model
#

import os
import cv2
import argparse
import numpy as np
import face2Dalign
#import face3Dfront
import datetime

#Dlib detector
lmarkDetector = "detect_config/shape_predictor_68_face_landmarks.dat"
#First align it
faceAlign = face2Dalign.Face2DAlign(lmarkDetector)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aligned Images.')
    parser.add_argument('-i', '--images', type=str, required=True, 
	                    help='List file of image location')
    parser.add_argument('-l', '--location', type=str, required=False, 
	                    help='Folder location of images')

    args = vars(parser.parse_args())
    image_list = args['images']

    with open(image_list, 'r') as f:
        img_names =  f.readlines()
    count = 0
    no_face = 0
    for name in img_names:
        print count
        #print name.split("/")
        svname =  "home/yinling/dbta/" + name.split("/")[4] + "/" + name.split("/")[5]
        print svname

        img = cv2.imread(name.strip())
        img = faceAlign.align(img)

        #if faceAlign.getLargestFace(img):
            #img = faceAlign.align(img)
        #else:
            #no_face += 1

        cv2.imwrite(svname, img)
        count += 1
        
    #print "No faces: {0}".format(no_face)
    print "Processed faces: {0}".format(count)
