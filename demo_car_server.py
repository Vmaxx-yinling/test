#!/usr/local/bin python
#Author: Yinling LIU
#Date: 2017.11.5

import libpysunergy
import cv2

imagelist = "/home/yinling/1010/datasets/vmmr.list"

#load yolo_coco detector
net1,names1 = libpysunergy.load("data/coco.data", "cfg/yolo.cfg", "yolo.weights")
threshold = 0.25
cfg_size = (608,608) #same as network input

#process images from the image list
with open(imagelist,'r') as f:
	img_names = f.readlines()
count = 0

for image in img_names:
	print image
	print count

	#load and convert image
	svname = "/home/yinling/1010/vmmrdb_crop/" + image.split("/")[5] +"_" + image.split("/")[-1]
	print svname
	img = cv2.imread(image.strip())
	(h,w,c) = img.shape
	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img_input = cv2.resize(img_rgb, cfg_size)

	#dets returns {py_names[class], prob, left, right, top, bot}
	dets = libpysunergy.detect(img_input.data, w, h, c, threshold, net1, names1)

	#crop cars from the maximun area of the bounding box if the detection belong to car, truck, and train
	obj_num = len(dets)
	area=[0]*obj_num
	detected = 0
	for i in range(0, obj_num):
		if dets[i][0] in ("car", "truck", "train"): 
			area[i] =(dets[i][5]-dets[i][4]) *(dets[i][3]-dets[i][2])	    
			detected += 1
	print area
	print detected, "car detected!"
	max_area = max(area)
	max_index = area.index(max_area)	
	img_result = img[dets[max_index][4]:dets[max_index][5],dets[max_index][2]:dets[max_index][3]].copy()
	cv2.imwrite(svname.strip(),img_result)
	count += 1

print "images processed:", count 
libpysunergy.free(net1)
