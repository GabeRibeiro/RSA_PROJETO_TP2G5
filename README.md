# RSA_PROJETO_TP2G5
PROJECT CLONED FROM https://code.nap.av.it.pt/atcll/object-detection-camera/-/tree/jetson_2gb WITH ONLY CHANGE MADE IN utils/visualization.py


* Changes made to support mqtt based message exchange about what objects are being detected.

# PREREQUISITES 
1\. FLASH JETSON NANO 2GB WITH JETPACK 4.6 LINK: [4.6 JETPACK IMAGE FOR JETSON NANO 2GB] (https://developer.nvidia.com/embedded/l4t/r32_release_v6.1/jeston_nano_2gb/jetson-nano-2gb-jp46-sd-card-image.zip)
2\. DO: sudo apt update
3\. DO NOT UPGRADE
4\. EXECUTE THE FOLLOWING COMMANDS:
 
      $ mkdir ${HOME}/project
      $ cd ${HOME}/project
      $ git clone https://github.com/jkjung-avt/jetson_nano.git
      $ cd jetson_nano
      $ ./install_basics.sh
      $ source ${HOME}/.bashrc
 
5\. CLONE THIS REPOSITORY 

      $ git clone git@github.com:GabeRibeiro/RSA_PROJETO_TP2G5.git
      
# HOW TO DEPLOY 
FOLLOW https://code.nap.av.it.pt/atcll/object-detection-camera/-/blob/jetson_2gb/yolov4/README.md

# HOW TO RUN (EXAMPLES)
1\. USB CAMERA

    $ cd yolov4
    $ python3 trt_yolo.py --usb 0 -m yolov4-416 --width 1280 --height 720

2\. VIDEO
  
    $ cd yolov4
    $ python3 trt_yolo.py --video ./vid2.mov -m yolov4-416
    
3\. IMAGE

    $ cd yolov4
    $ python3 trt_yolo.py --image ./dog.jpg -m yolov4-416
