"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse
import json
import threading
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import logging
import paho.mqtt.client as mqtt
import uuid
import numpy as np
import scipy.stats as st

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from jproperties import Properties

logging.basicConfig(level=logging.INFO)
np.random.seed(1)

CAMERA_COORDINATES = (0, 0)
WINDOW_NAME = 'TrtYOLODemo'
CoordinatesFileName = "posicoes4.txt"
FileXMAX = 1600
FileXMIN = 640
FileYMAX = 910
FileYMIN = 390

def confidence_interval(data, confidence=0.95):
	if len(data) < 2: 
		return (0, 0)
	sampleMean = np.mean(data)         #sample mean 
	sampleStandardError = st.sem(data) #sample standard error
	#create 95% confidence interval for the population mean
	confidenceInterval = st.norm.interval(alpha=confidence,loc=sampleMean,scale=sampleStandardError)
	
	#print the 95% confidence interval for the population mean
	#print('The 95% confidence interval for the population mean weight :',confidenceInterval)

	# to change the tuple
	confidenceInterval_as_list = list(confidenceInterval)
	if np.isnan(confidenceInterval[0]):
		confidenceInterval_as_list[0] = 0

	if np.isnan(confidenceInterval[1]):
		confidenceInterval_as_list[1] = 0

	return tuple(confidenceInterval_as_list)

def parse_args():
	"""Parse input arguments."""
	desc = ('Capture and display live camera video, while doing '
			'real-time object detection with TensorRT optimized '
			'YOLO model on Jetson')
	parser = argparse.ArgumentParser(description=desc)
	parser = add_camera_args(parser)
	parser.add_argument(
		'-c', '--category_num', type=int, default=80,
		help='number of object categories [80]')
	parser.add_argument(
		'-m', '--model', type=str, default="yolov3-416",
		help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
			  '[{dimension}], where dimension could be a single '
			  'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
	parser.add_argument(
		'-r', '--rotate', type=int, default=0,
		help='rotate angle (0, 90, 180, 270)')
	parser.add_argument(
		'-l', '--location', nargs=2, type=float, default=(0.0,0.0),
		help='camera location where location is in lat long format'
			  '(e.g. 40 8)')
	parser.add_argument(
		'-u', '--use_properties', type=str, default="",
		help='Use properties file'
		)
	args = parser.parse_args()
	
	if args.use_properties is not "":
		logging.info("Use Properties File")

		configs = Properties()
		with open(args.use_properties, 'rb') as config_file:
			configs.load(config_file)

		logging.info("Use Properties File with " + str(configs.items()))
		args.camera_id = int(configs['CAMERA_ID'].data)
		args.rtsp = configs['CAMERA_RTSP'].data        
		args.location = (configs['CAMERA_LATITUDE'].data, configs['CAMERA_LONGITUDE'].data)
		args.zoom = configs['CAMERA_ZOOM'].data
		args.heading = configs['CAMERA_HEADING'].data
		args.rotate = configs['CAMERA_ROTATION'].data    
		args.rtsp_latency = int(configs['CAMERA_RTSP_LATENCY'].data)
		args.width = int(configs['CAMERA_RTSP_WIDTH'].data) 
		args.height = int(configs['CAMERA_RTSP_HEIGHT'].data)
		args.model = configs['DETECTION_MODEL'].data    
		args.detection_x_add = float(configs['DETECTION_ADD_X_COORDINATES'].data)
		args.detection_y_add = float(configs['DETECTION_ADD_Y_COORDINATES'].data)
	
	logging.info(args)
	return args

def parseCoordinates(estrutura):
	f = open(CoordinatesFileName, "r")

	lines = f.readlines()
	for l in lines:
		if len(l) != 1:
			line = l.strip()
			fields = line.split(" ")
			if fields[2] not in estrutura:
				estrutura[fields[2]] = [{"ymax": int(fields[3]), "lat": float(fields[4]), "lon" : float(fields[5])}]
			else:
				estrutura[fields[2]].append({"ymax": int(fields[3]), "lat": float(fields[4]), "lon" : float(fields[5])})
				estrutura[fields[2]] = sorted(estrutura[fields[2]], key=lambda d: d["ymax"])

def findCoordinates(xs_shortcut, estrutura, x, y):
	return CAMERA_COORDINATES #(40.634823, -8.660173)

	##for e in xs_shortcut:
	##    if x <= e:
	##        for sq in estrutura[str(e)]:
	##            if y <= sq["ymax"]:
	##                return (sq["lat"], sq["lon"])


def loop_and_detect(cam, trt_yolo, conf_th, vis, args=None):
	"""Continuously capture images from camera and do object detection.

	# Arguments
	  cam: the camera instance (video source).
	  trt_yolo: the TRT YOLO object detector instance.
	  conf_th: confidence/score threshold for object detection.
	  vis: for visualization.
	"""
	client = mqtt.Client()
	client.connect("127.0.0.1", 1883, 60)

	mqtt_th = threading.Thread(target=client.loop_forever)
	mqtt_th.start()

	struct = {}
	parseCoordinates(struct)
	xs = sorted(set(struct), key=lambda s: int(s))
	xs = [int(e) for e in xs]


	#last = time.time()
	full_scrn = False
	sum_times = 0
	fps = 0.0
	tic = time.time()

	#last frame had no detections
	last_no_detect = True
	last_no_detect_t = True

	#Logging variables
	log_detection_count = 0

	rotation_code = args.rotation_code

	while True:#for number in range(0, 200):
		#if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
			#break
		img = cam.read()
		if rotation_code is not None:
			img = cv2.rotate(img, rotation_code)

		if log_detection_count == 0:
			#print("Log - Got a new capture from the camera - "+str(time.time()))
			logging.info("Log - Got a new capture from the camera - " + str(time.time()))
			
		log_detection_count = ((log_detection_count + 1) % 300)
		#drawlines(img)
		#print(img.shape[:2])
		t = time.time()
		if img is None:
			break
		start = time.time()
		boxes, confs, clss = trt_yolo.detect(img, conf_th)
		ps = []
		others = []

		confidences = {}
		# Transform class indexes into names
		names = [vis.get(int(class_index), 'CLS{}'.format(int(class_index))) for class_index in clss]
		
		for ind in range(0, len(clss)):	
			# Construct list with confidence of all detections
			name = vis.get(int(clss[ind]), 'CLS{}'.format(int(clss[ind])))
			
			if confidences.get(name) is None:
				confidences[name] = []

			confidences[name].append(confs[ind])
			
			if int(clss[ind]) == 0:
				dic = {
						"id" : str(uuid.uuid4()), #int(clss[ind])),
						"label" : "Person",
						"confidence" : str(confs[ind]),
						"bbbox" : [[int(boxes[ind][0]), int(boxes[ind][1])], 
								   [int(boxes[ind][2]-boxes[ind][0]), int(boxes[ind][3]-boxes[ind][1])]],
						"coordinates" : {}
						}

				point_x = int(boxes[ind][0]) + (int(boxes[ind][2]) - int(boxes[ind][0]))/2
				point_y = int(boxes[ind][3]) - 5

				#if (point_x > FileXMIN and point_x <= FileXMAX and point_y > FileYMIN and point_y <= FileYMAX):
				(lat, lon) = findCoordinates(xs, struct, point_x, point_y)
				dic["coordinates"]["lat"] = str(float(lat) + args.detection_x_add)
				dic["coordinates"]["lon"] = str(float(lon) + args.detection_y_add)

				ps.append(dic)
			else:
				name = vis.get(int(clss[ind]), 'CLS{}'.format(int(clss[ind])))
				if name == "person" or name == "car" or name == "truck":
					dic = {
							"id" : str(uuid.uuid4()),
							"label" : name,
							"confidence" : str(confs[ind]),
							"bbbox" : [[int(boxes[ind][0]), int(boxes[ind][1])],
									[int(boxes[ind][2]-boxes[ind][0]), int(boxes[ind][3]-boxes[ind][1])]],
							"coordinates" : {}
							}
					others.append(dic)

		detected = False
		if len(ps) != 0:
			detected = True

		final_json_p = {
						"detectedPerson" : str(detected),
						"listOfPeople"   : ps,
						"timestamp"      : str(time.time()),
						"heading"        : str(args.heading),
						"location"       : {"lat": CAMERA_COORDINATES[0], "lon": CAMERA_COORDINATES[1]}
					}

		#print(final_json_p)
		client.publish("Jetson/Camara/Objects/People", payload=json.dumps(final_json_p), qos=0, retain=False)
		others.extend(ps)

		final_json_p = {
						"detectedPerson" : str(detected),
						"listOfObjects"   : others,
						"timestamp"      : str(time.time()),
						"heading"        : str(args.heading),
						"location"       : {"lat": CAMERA_COORDINATES[0], "lon": CAMERA_COORDINATES[1]}
					}

		client.publish("Jetson/Camara/Objects/Total", payload=json.dumps(final_json_p), qos=0, retain=False)
		
		count_of_classes = {}
		avg_confidences_of_classes = {}
		interval_confidences_of_classes = {}

		#logging.info("Classes={} | names={} | confs={} | confidences={}".format(clss, names, confs, confidences))
		
		#count_of_classes={} | confidences_of_classes={}
		for item in names:
			count_of_classes[item] = count_of_classes.get(item, 0) + 1
			
			
		
			values = confidences[item]
			avg_confidences_of_classes[item] = sum(values) / len(values)
			interval_confidences_of_classes[item] = confidence_interval(values)

		for class_label in count_of_classes:
			final_json_p = {
			 			"class_label"    : class_label,
			 			"class_count"	 : count_of_classes[class_label],
			 			"confidence_avg" : avg_confidences_of_classes[class_label],
			 			"confidence_min_interval"	 : interval_confidences_of_classes[class_label][0],
			 			"confidence_max_interval"	 : interval_confidences_of_classes[class_label][1],
			 			"timestamp"      : str(time.time()),
			 			"heading"        : str(args.heading),
			 			"zoom"			 : str(args.zoom),
			 			"location"       : {"lat": str(float(CAMERA_COORDINATES[0]) + float(args.detection_x_add)), "lon": str(float(CAMERA_COORDINATES[1]) + float(args.detection_y_add))}
}
			final_json_new_format = {
						"timestamp"      : time.time(),
						"classLabel"    : class_label,
						"classCount"	 : count_of_classes[class_label],
						"confidenceAvg" : avg_confidences_of_classes[class_label],
						"confidenceMinInterval"	 : interval_confidences_of_classes[class_label][0],
						"confidenceMaxInterval"	 : interval_confidences_of_classes[class_label][1],
						"heading"        : float(args.heading),
						"zoom"			 : float(args.zoom),
						"latitude": float(float(CAMERA_COORDINATES[0]) + float(args.detection_x_add)), 
						"longitude": float(float(CAMERA_COORDINATES[1]) + float(args.detection_y_add)),
						"cameraID": args.camera_id,
						"test": str({}),
						"algorithm": args.model,
					}

			#logging.info(final_json_p)
			client.publish("Jetson/Camara/Count", payload=json.dumps(final_json_p), qos=0, retain=False)
			client.publish("jetson/camera/count", payload=json.dumps(final_json_new_format), qos=0, retain=False)



		#print(time.time()-last)
		end = time.time()
		#print(end-start)
		#img = vis.draw_bboxes(img, boxes, confs, clss)
		#img = show_fps(img, t)
		#cv2.imshow(WINDOW_NAME, img)
		toc = time.time()
		curr_fps = 1.0 / (toc - tic)
		# calculate an exponentially decaying average of fps number
		fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
		tic = toc
		key = cv2.waitKey(1)
		if key == 27:  # ESC key: quit program
			break
		#elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
			#full_scrn = not full_scrn
			#set_display(WINDOW_NAME, full_scrn)

	#print("Detection rate for 2 min: {}".format(sum_times/1000.0))

def drawlines(frame):
	cv2.rectangle(frame,(0,0),(600,1080),(0,0,0),-1)

def main():
	global CAMERA_COORDINATES
	
	args = parse_args()
	if args.category_num <= 0:
		raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
	if not os.path.isfile('yolo/%s.trt' % args.model):
		raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

	cam = Camera(args)
	if not cam.isOpened():
		raise SystemExit('ERROR: failed to open camera!')

	cls_dict = get_cls_dict(args.category_num)
	logging.info("cls_dict" + str(cls_dict))
	yolo_dim = args.model.split('-')[-1]
	if 'x' in yolo_dim:
		dim_split = yolo_dim.split('x')
		if len(dim_split) != 2:
			raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)
		w, h = int(dim_split[0]), int(dim_split[1])
	else:
		h = w = int(yolo_dim)
	if h % 32 != 0 or w % 32 != 0:
		raise SystemExit('ERROR: bad yolo_dim (%s)!' % yolo_dim)

	trt_yolo = TrtYOLO(args.model, (h, w), args.category_num)


	#open_window(
		#WINDOW_NAME, 'Camera TensorRT YOLO Demo',
		#cam.img_width, cam.img_height)
	#vis = BBoxVisualization(cls_dict)

	#Define if rotation is needed
	args.rotation_code = None

	#print("\n\n\n\nARGS ROTATE" + str(args.rotate))
	if args.rotate == 0:
		args.rotation_code = None
		logging.info("Log - Rotation code: No rotation")
	elif args.rotate == 90:
		args.rotation_code = cv2.ROTATE_90_CLOCKWISE
		logging.info("Log - Rotation code: cv2.ROTATE_90")
	elif args.rotate == 180:
		args.rotation_code = cv2.ROTATE_180
		args.logging.info("Log - Rotation code: cv2.ROTATE_180")
	elif args.rotate == 270:
		args.rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
		logging.info("Log - Rotation code: cv2.ROTATE_90_COUNTERCLOCKWISE")
	
	#print("\n\n\n\nARGS ROTATE CODE" + str(rotation_code) + str(type(rotation_code)))
	
	# Parse camera coordinates
	CAMERA_COORDINATES = args.location
	args.zoom = args.zoom
	args.heading = args.heading
	logging.info("Log - Camera Coordinates={}, Heading={}, Zoom={}".format(str(CAMERA_COORDINATES), str(args.zoom), str(args.heading)))

	loop_and_detect(cam, trt_yolo, conf_th=0.6, vis=cls_dict, args=args)

	cam.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
