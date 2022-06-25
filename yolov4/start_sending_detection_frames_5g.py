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
server_addr = "10.0.20.44"#"193.136.93.46"#"10.0.20.44"#"193.136.93.46"#"10.0.20.44"#"10.0.20.44"#"193.136.93.46" #"10.0.20.44"
server_port = 9000;

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
    ip = "10.0.20.44"#"193.136.93.46" #"10.0.20.44" #ip = "193.136.93.46"
    port = 9000;
    cs_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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

    while True:
        #msg = cs_server.recv(4096)
        #print("Full cycle time {}\n".format(time.time()-inicio))
        print("Start")
        selected = pickle.dumps(lastframe)
        size = len(selected)
        p = struct.pack('I', size)
        selected = p + selected
        inicio = time.time()
        print(len(selected))
        print("Frame Started Sent: {}".format(time.time()))
        pos = 0
        ended = False
        t_total = 0
        while not ended: #512 #1024 #2048 #4096 #8192 #16384 32768 64000
            if pos+32768 < len(selected):
                s_msg = selected[pos:pos+32768]
                pos += 32768
            else:
                s_msg = selected[pos:]
                ended = True
            t_s = time.time()
            cs_server.sendto(s_msg, (server_addr, server_port))
            t_total += (time.time()-t_s)
        
        
        print(t_total)
        time.sleep(60)


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

