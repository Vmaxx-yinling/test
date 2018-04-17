# coding: utf-8


'''
this script crop all the images with default face detection
and built darknet-readble labels for AffectNet datasets.
'''

import os
import csv
import cv2
import scipy.misc

def generate_emlabel(emotion):
    if emotion == '0':
        emlable = 'neutral'
    if emotion == '1':
        emlable = 'happy'
    if emotion == '2':
        emlable = 'sad'
    if emotion == '3':
        emlable = 'surprise'
    if emotion == '4':
        emlable = 'fear'
    if emotion == '5':
        emlable = 'disgust'
    if emotion == '6':
        emlable = 'angry'
    if emotion == '7':
        emlable = 'contempt'
    if emotion == '8':
        emlable = 'none'
    if emotion == '9':
        emlable = 'uncertain'
    if emotion == '10':
        emlable = 'noface'        
    return emlable


id = 1
with open('valid_lable.list','rb') as datafile:
    datareader = csv.reader(datafile, delimiter = ' ')
    #headers = datareader.next()
    for row in datareader:
        short_name = row[0]
        full_name = "/videos/affectnet/Manually_Annotated/Manually_Annotated/affectnet_oriimg/" + short_name
        #print full_name
        img = cv2.imread(full_name.strip())
        (h0,w0,c0)= img.shape
        #print h0,w0
        emotion =  row[1]
        left = int(row[2])
        top = int(row[3])
        right = int(row[2]) + int(row[4])
        bottom = int(row[3]) + int(row[5])
        #print left,right,top,bottom
        faceimg = img[top:bottom,left:right].copy()
        emlable = generate_emlabel(emotion)
        save_name = "/videos/affectnet/cvface/valid/" + emlable + '_' + str(id) + ".jpg"
        cv2.imwrite(save_name,faceimg)
        id += 1
        if id % 100 == 0:
            print id 





