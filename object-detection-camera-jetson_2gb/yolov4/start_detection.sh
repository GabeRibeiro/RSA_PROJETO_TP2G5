#!/bin/bash


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd $SCRIPT_DIR

python3 start_detection_json.py --use_properties camera.properties

#rtsp rtsp://username:password@atcll-p22-camera.nap.av.it.pt --rtsp_latency 1 --height 1920 --width 1080 -m yolov3-416 -l 40.63972 -8.64352 -r 0
