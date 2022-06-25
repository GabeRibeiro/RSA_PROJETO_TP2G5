"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""
import socket
import struct
import threading
import pickle
import numpy as np


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization

from utils.yolo_with_plugins import TrtYOLO

##global variables

jetsonID = 0
cs_server = None

lastframe = None

existPersonApp = False

WINDOW_NAME = 'TrtYOLODemo'
   

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
        '-m', '--model', type=str, required=True,
        help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'))
    args = parser.parse_args()
    return args


##port listeners threads

def frame_listener_thread(cs_server, cam, trt_yolo, conf, vis):
    global lastframe

    loop_and_detect(cs_server, cam, trt_yolo, conf_th=conf, vis=vis)



def loop_and_detect(socket, cam, trt_yolo, conf_th, vis):
    global lastframe
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    teste = ["0", "0", "0", "0", "0", "0", "0", "0",\
             "1", "1", "1", "1", "1", "1", "1", "1",\
             "0", "0", "1", "1", "0", "0", "1", "1"]
    teste_idx = 0

    cs_server = socket
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        img = cam.read()
        if img is None:
            break

        lastframe = img





def main():
    #Connect to result agregator
    #ip = "10.0.20.44" #ip = "193.136.93.46"
    #port = 9000;
    #cs_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #cs_server.connect((ip, port))

    time.sleep(0.20)

    print("Ready")

    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    cls_dict = get_cls_dict(args.category_num)
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
    #    WINDOW_NAME, 'Camera TensorRT YOLO Demo',
    #    cam.img_width, cam.img_height)
    vis = BBoxVisualization(cls_dict)
    reading_module = threading.Thread(target=frame_listener_thread, args=[cs_server, cam, trt_yolo, 0.3, vis])
    reading_module.setDaemon(True)
    reading_module.start()
    inicio = 0

    dir = "yolo"
    frames = []


    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([dir, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([dir, "yolov3-416.weights"])
    configPath = os.path.sep.join([dir, "yolov3-416.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    while True:
        image = lastframe
        (H, W) = image.shape[:2]
            

        # determine only the *output* layer names that we need from YOLO

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > 0.4:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                0.3)

         # ensure at least one detection exists
        existPerson = False
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                #test if a person exists
                if (LABELS[classIDs[i]] == "person" and confidences[i] > 0.40):
                    existPerson = confidences[i]
                    break
                    
        print("confidence: "+str(existPerson))
        print(time.time()-inicio)



    cam.release()
    cv2.destroyAllWindows()
    '''
    teste = ["0", "0", "0", "0", "0", "0", "0", "0",\
             "1", "1", "1", "1", "1", "1", "1", "1",\
             "0", "0", "1", "1", "0", "0", "1", "1"]
    teste_idx = 0

    while True:
        cs_server.sendall((str(jetsonID)+" "+teste[teste_idx]).encode("UTF-8"))
        teste_idx = (teste_idx+1) % 24
        time.sleep(0.15)
    '''


if __name__ == '__main__':
    main()
