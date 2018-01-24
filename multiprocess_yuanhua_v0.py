#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import cv2
import os
import dlib
import glob
import numpy as np
import time
#import libpysunergy

import threading
import multiprocessing
import socket
import json
import os.path
#from lshash import LSHash
#import fcntl


top = 1
scale = 4
#for 8G GPU memory thread_num = 5
#changed to 1 to avoid cuda out of memory error
thread_num = 1

def enroll_built(faces_enroll_path):
    descriptors = []
    count = 0
    enrolled_number = 0
	#order images at which entries appear in the filesystem
    for f in glob.glob(os.path.join(faces_enroll_path, "*.jpg")):
        #print f
        img = cv2.imread(f)
        (ho,wo,co) = img.shape
        dets = detector(img, 1)
        #print("{} image: {}, faces detected: {}".format(count, f, len(dets)))
        #faces = dlib.full_object_detections()
        count += 1
        for i, d in enumerate(dets):
            [x0,x1,y0,y1] = [max(d.left(),0),
                             min(d.right(),wo-1),
                             max(d.top(),0),
                             min(d.bottom(),ho-1)]
            faceimg = img[y0:y1, x0:x1].copy()
            #cv2.rectangle(img,(x0,y0),(x1,y1),(0, 255, 0),5)
            (h,w,c) = faceimg.shape
            sv_name = "./enrolled_yuanhua/" + str(enrolled_number) + ".jpg"
            enrolled_number += 1
            print "enroll person: {}".format(enrolled_number)
            cv2.imwrite(sv_name,faceimg)   

            shape = sp(img, d)        
            #faces.append(shape)
            #image = dlib.get_face_chip(img, faces[i], size=224, padding=0)
            #faceimgalign = image.copy()
            #sv_align = "./enrolled/" + str(enrolled_number) + "_align.jpg"
            #cv2.imwrite(sv_align,faceimgalign)
            
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            # convert to numpy array
            v = np.array(face_descriptor)
            #print v
            #descriptors.append(v)
            #time.sleep(3)
        #cv2.imshow(str(f),img)
        #cv2.waitKey(0)
    np.savetxt('enrolled_built_yuanhua.fea',descriptors)
    print "Total {} people were enrolled in system".format(enrolled_number)
    return descriptors


def fea_compare(pic):
    result = {}
    try:
		imageid = pic.split('/')[-1].split('.')[0]

		try:
		    img_test = cv2.imread(pic)
		    (ht,wt,ct) = img_test.shape
		    #cv2.imshow("img_test",img_test)
		    #cv2.waitKey(0)
		except:
			return json.dumps(result, sort_keys=True)

		dets_test = detector(img_test, 1)
		#print("faces detected: {}".format(len(dets_test)))

		dists = []
		ds_test = []
		result_list = []
		descriptors = np.loadtxt('enrolled_built_yuanhua.fea')

        t0 = time.time()
		for k, d in enumerate(dets_test):       
			#get the  bounding box
			[x0,x1,y0,y1] = [max(d.left(),0),
							 min(d.right(),wt-1),
							 max(d.top(),0),
							 min(d.bottom(),ht-1)]
			faceimg = img_test[y0:y1, x0:x1].copy() 
			(h,w,c) = faceimg.shape
			
			temp_path = 'temp/' + imageid +  "_" + str(k) + '.jpg'
			#tempname = imageid + "_" + str(k) + '.jpg'
			cv2.imwrite(temp_path, faceimg)

			#calculate the descriptor       
			shape_test = sp(img_test, d)
			face_descriptor_test = facerec.compute_face_descriptor(img_test, shape_test)
			d_test = np.array(face_descriptor_test)
			ds_test.append(d_test)
			
			t1 = time.time()
			#calculate the euclidean distance
			for i in descriptors:
				dist = np.linalg.norm(i - d_test)
				dists.append(dist)
				#print dist

			#make a dict using candidate and distance 
			candidates = range(len(descriptors))
			c_d = dict(zip(candidates, dists))
            
			t2 = time.time()
			#find if any match using threshhold 0.5
			threshold = 0.50
			cd_sorted = sorted(c_d.iteritems(), key=lambda d: d[1])
			mindis = cd_sorted[0][1]

			#print cd_sorted[0:3]
			if mindis < threshold:
				match = True
				matchid = cd_sorted[0][0]
				#print "Matched id {}: with euclidean distance: {}".format(matchid,mindis)
				#match_img = "./enrolled/" + str(cd_sorted[0][0]) + '.jpg'
				#print match_img
				#matchimg = cv2.imread(match_img)     
				#cv2.imshow("matched", matchimg)
				#cv2.waitKey(0)
				conf = abs(threshold - mindis)/float(threshold)

			else:
				#print "NO match found"
				#descriptors.append(d_test)
				match = False
				matchid = None
				conf = abs(mindis - threshold)/float(threshold)

            result_list.append({"match": match,"matchid": matchid,"conf_score": conf}
			                    "box_left": x0, "box_right": x1, "box_top": y0, "box_bottom": y1 )
            
		t3 = time.time()
		print "descriptor:{}, distance:{}, sorted:{}".format(str(t1-t0), str(t2-t2),str(t3-t2))

		result["faces"] = result_list
		return json.dumps(result, sort_keys=True)

    except Exception, e:
		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
		return json.dumps(result, sort_keys=True)			


'''
root = './'
fea_root = root + 'feas/'
fea_thresh = 10000

#extractor input size
fw = 125
fh = 140
fc = 1

def fea_decode(filename):
	feas = []
	f = open(filename,'r')
	while(1):
		line = f.readline()
		if not line:
			break
		temp = []
		s = line.split()
		for i in range(0, len(s)):
			temp.append(round(float(s[i]),2))
		feas.append(temp)
	#f.close()
	return feas

def fea_write(filename, feas):
	f = open(filename, 'w')
	for i in range(0, len(feas)):
		for j in range(0, len(feas[i])):
			f.write(str(feas[i][j]) + ' ')
		f.write('\n')
		f.flush()
	#f.close()

def fea_compare(base, fea):
	#print fea
	lsh = LSHash(30,4096,10)
	for i in range(0,len(fea)):
		fea[i]=round(float(fea[i]),2)

	#load base features
	base_fea = fea_decode(base)
	for i in range(0, len(base_fea)):
		lsh.index(base_fea[i])

	q = lsh.query(fea, distance_func = 'euclidean')

	if len(q) == 0:
		base_fea.append(fea)
		fea_write(base,base_fea)
		return False,1.0

	mindis = q[0][1]
	if mindis<fea_thresh:
		conf = abs(fea_thresh - mindis)/float(fea_thresh)
		return True, conf
	else:
		base_fea.append(fea)
		fea_write(base,base_fea)
		conf  = abs(mindis - fea_thresh)/float(mindis)
		return False, conf
'''


class Server(object):
	def __init__(self, hostname, port):
		self.hostname = hostname
		self.port = port

	def start(self):
		print "server start"
		#detector = dlib.get_frontal_face_detector()

		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.bind((self.hostname, self.port))
		self.socket.listen(thread_num)

		print "start to create threads:"
		for i in range(0, thread_num):
			print i
			#process = multiprocessing.Process(target=service, args=(self.socket, detector))
			process = multiprocessing.Process(target=service, args=(self.socket))
			process.daemo = True
			process.start()

#def service(s,detector):
def service(s):
	'''
	#sunergy classifier
	net1,names1 = libpysunergy.load(root + "data/age1.1.data", root + "cfg/age1.1.cfg", root + "weights/age1.1.weights")
	net2,names2 = libpysunergy.load(root + "data/gender1.1.data", root + "cfg/gender1.1.cfg", root + "weights/gender1.1.weights")
	extractor, names3 = libpysunergy.load(root + "data/face_extractor.data", root + "cfg/face_extractor.cfg", root + "weights/face_extractor.weights")
    '''

	while(1):
		conn, address = s.accept()
		buff = conn.recv(16384)
		print buff
		msg = json.loads(buff)
		#eventname = msg['eventname']
		pic = msg["imagename"]

		#run models: face/landmark detection, and face recognition     
        predictor_path = "shape_predictor_5_face_landmarks.dat"
        face_rec_model_path = "face_recognition_resnet_model_v1.dat"
        detector = dlib.get_frontal_face_detector()
        sp = dlib.shape_predictor(predictor_path)
        facerec = dlib.face_recognition_model_v1(face_rec_model_path)

		#t0 = time.time()
		#first build the enroll system
		#descriptors = enroll_built("/home/yinling/face_recg/yuanhua/dmimg_cp/")
		#t1 = time.time()

		#now deal with the test person
		#testimg = "/home/yinling/face_recg/yuanhua/dmimg_cp/1012879TTZJDMBL2017090201.jpg"
		#testimg = "/home/yinling/face_recg/yl.jpeg"
		#img= cv2.imread(testimg)
		#cv2.imshow("img_test",img)
		#cv2.waitKey(0)

		#rj  = predict(net1, net2, names1, names2, extractor, detector,eventname, pic)        
		rj = fea_compare(pic)
        print rj
        conn.sendall(rj)
        conn.close()
		#t2 = time.time()

	for process in multiprocessing.active_children():
		#process.terminate()
		process.join()

'''
def predict(net1, net2, names1, names2, extractor, detector, eventname, pic):
	result ={}
	try:
		top = 1
		imageid = pic.split('/')[-1].split('.')[0]
		
		#1 read image from opencv
		try:
			im = cv2.imread(pic)

			#2 detector face using face detector
			(ow,oh,oc) = im.shape
			im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			im_resize = cv2.resize(im_grey,(oh/scale,ow/scale))
		except:
			return json.dumps(result, sort_keys=True)

		faces = detector(im_resize,1)

		facenum = len(faces)

		result_list = []

		t0 = time.time()
		for i in range(0,facenum):
			[x0,x1,y0,y1] = [faces[i].left()*scale,
							faces[i].right()*scale,
							faces[i].top()*scale,
							faces[i].bottom()*scale]
			x0 = max(x0,0)
			y0 = max(y0,0)
			x1 = min(x1, oh-1)
			y1 = min(y1,ow-1)

			faceimg = im[y0:y1, x0:x1].copy()
			
			(w,h,c) = faceimg.shape
			#get result from sunergy
			dets1 = libpysunergy.predict(faceimg.data, w, h, c, top, net1, names1)
			age = int(str(dets1[0][0]))
			age_prob = dets1[0][1]
			if float(age_prob) < 0.5:
				age_prob = 1 - float(age_prob)

			dets2 = libpysunergy.predict(faceimg.data, w, h, c, top, net2, names2)
			gender = dets2[0][0]
			gender_prob = dets2[0][1]
			if float(gender_prob) < 0.5:
				gender_prob = 1 - float(gender_prob)

			p_x0 = max(x0-10,0)
			p_y0 = max(y0-10,0)
			p_x1 = min(x1+10,oh-1)
			p_y1 = min(y1+10,ow-1)

			p_faceimg = im[p_y0:p_y1, p_x0:p_x1].copy()

			#extract face feature
			p_faceimg_gray = cv2.cvtColor(p_faceimg,cv2.COLOR_BGR2GRAY)
			p_faceimg_gray = cv2.resize(p_faceimg_gray,(fw,fh))
			#print p_faceimg_gray
			feat = libpysunergy.extract_batch(p_faceimg_gray.data, fw, fh, 1, 1, extractor)
			#feat = libpysunergy.extract(p_faceimg_gray.data, fw, fh, 1, extractor)
			base = fea_root + eventname + '.fea'
			if not os.path.isfile(base):
				bf = open(base,'w')
				bf.close()
			#check lock
			lock = open(fea_root + eventname + '.lock','w')
			fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

			seen, seen_prob  = fea_compare(base, feat)

			#release lock
			lock.close()

			temp_path = 'temp/' + imageid +  "_" + str(i) + '.jpg'
			tempname = imageid + "_" + str(i) + '.jpg'
			cv2.imwrite(temp_path, p_faceimg)
			#get result from ms
			#s3.upload_file(temp_path,"vmaxx1","agel_ms/" + tempname , ExtraArgs={'ACL': 'public-read'})
			#(ms_age,ms_gender) = ms_result('https://s3-us-west-2.amazonaws.com/vmaxx1/agel_ms/' + tempname)

			#if ms_age > 0:
				#age = int(0.6336586*ms_age + 0.39211679*age - 0.617975925)
			#if (gender == 'male' and ms_gender == 'female') or (gender == 'female' and ms_gender == 'male'):
				#gender = 'female'

			#write to json
			
			age_prob = round(age_prob, 3)
			gender_prob = round(gender_prob, 3)
			seen_prob = round(seen_prob, 3)
			
			result_list.append({"age": age, "age_prob": age_prob, "gender": gender, 
							   "gender_prob": gender_prob, "seen": seen, "seen_prob": seen_prob,
								"box_left": x0, "box_right": x1, "box_top": y0, "box_bottom": y1})
		t1 = time.time()
		print "detection time:" + str(t1-t0) + "s"	
			
		result["faces"] = result_list
		return json.dumps(result, sort_keys=True)

	except Exception, e:
		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
		return json.dumps(result, sort_keys=True)
'''


if __name__ == '__main__':

	#create socket, listen to port
	private_ip = socket.gethostbyname(socket.getfqdn())
	server = Server(private_ip,101)
	#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	#s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	#s.bind((private_ip, 101))
	#s.listen(5)
	server.start()
	
