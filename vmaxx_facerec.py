#Age&Gender prediction API demo
#Vmaxx Inc. 2017

#modules to import
import requests
from poster.encode import multipart_encode
from poster.streaminghttp import register_openers
import urllib2
import timeit

register_openers()

#replace 'test.jpg' with your filename
#"task":'analysis' are required and should not be modified
f = open("yuanhua.list")

while 1:
	start_time = timeit.default_timer()
	line = f.readline()
	if not line:
		break
	line = line.strip('\n')
	datagen, headers = multipart_encode({"filetoupload": open(line, "rb"), 'task': 'analysis', 'eventname':'my1030de4'})
	#datagen, headers = multipart_encode({"filetoupload": open(line, "rb"), 'task': 'analysis'})

	#service is locating at 54.201.44.70/face.php
	request = urllib2.Request("http://54.186.163.50/face.php", datagen, headers)
	#request = urllib2.Request("http://192.168.1.189/face.php", datagen, headers)

	#execute and print responed json-formated result
	print 'image_name: ' + line	
	print urllib2.urlopen(request).read()
	print timeit.default_timer() - start_time, "seconds"	


