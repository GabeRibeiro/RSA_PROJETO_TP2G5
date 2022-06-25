from cv2 import line, VideoCapture, imwrite, waitKey, IMWRITE_JPEG_QUALITY,imshow
from datetime import datetime
from os import environ, chdir, system
from os import path as p
import logging as log
from json import load
import threading
import time
import pickle
import socket 

f = open("StartFrame", "rb")
data = b""
aux = f.read(16384)
while(aux):
    data = data + aux
    aux = f.read(16384)
    
f.close()
frame = pickle.loads(data)
line(frame, (0,390), (1800,390), (255,0,255), 1)
#line(frame, (0,420), (1800,420), (255,0,255), 1)
line(frame, (0,460), (1800,460), (255,0,255), 1)
#line(frame, (0,510), (1800,510), (255,0,255), 1)
line(frame, (0,570), (1800,570), (255,0,255), 1)
#line(frame, (0,640), (1800,640), (255,0,255), 1)
line(frame, (0,720), (1800,720), (255,0,255), 1)
#line(frame, (0,810), (1800,810), (255,0,255), 1)
line(frame, (0,910), (1800,910), (255,0,255), 1)
#line(frame, (0,1020), (1800,1020), (255,0,255), 1)

#line(frame, (80,0), (80,1200), (255,0,255), 1)
#line(frame, (160,0), (160,1200), (255,0,255), 1)
#line(frame, (240,0), (240,1200), (255,0,255), 1)
#line(frame, (320,0), (320,1200), (255,0,255), 1)
#line(frame, (400,0), (400,1200), (255,0,255), 1)
#line(frame, (480,0), (480,1200), (255,0,255), 1)
#line(frame, (560,0), (560,1200), (255,0,255), 1)
line(frame, (640,0), (640,1200), (255,0,255), 1)
line(frame, (720,0), (720,1200), (255,0,255), 1)
line(frame, (800,0), (800,1200), (255,0,255), 1)
line(frame, (880,0), (880,1200), (255,0,255), 1)
line(frame, (960,0), (960,1200), (255,0,255), 1)
line(frame, (1040,0), (1040,1200), (255,0,255), 1)
line(frame, (1120,0), (1120,1200), (255,0,255), 1)
line(frame, (1200,0), (1200,1200), (255,0,255), 1)
line(frame, (1280,0), (1280,1200), (255,0,255), 1)
line(frame, (1360,0), (1360,1200), (255,0,255), 1)
line(frame, (1440,0), (1440,1200), (255,0,255), 1)
line(frame, (1520,0), (1520,1200), (255,0,255), 1)
line(frame, (1600,0), (1600,1200), (255,0,255), 1)
#line(frame, (1680,0), (1680,1200), (255,0,255), 1)
#line(frame, (1760,0), (1760,1200), (255,0,255), 1)

imshow('frame',frame)
waitKey(0)
