# RSA_PROJETO_TP2G5
* PROJECT CLONED FROM https://code.nap.av.it.pt/atcll/object-detection-camera/-/tree/jetson_2gb WITH ONLY CHANGE MADE IN utils/visualization.py


* Changes made to support mqtt based message exchange about what objects are being detected.

# PREREQUISITES 
1\. FLASH JETSON NANO 2GB WITH JETPACK 4.6 LINK: [4.6 JETPACK IMAGE FOR JETSON NANO 2GB] (https://developer.nvidia.com/embedded/l4t/r32_release_v6.1/jeston_nano_2gb/jetson-nano-2gb-jp46-sd-card-image.zip)

2\. DO:  

      $ sudo apt update
      
3\. DO NOT UPGRADE

4\. EXECUTE THE FOLLOWING COMMANDS:
 
      $ mkdir ${HOME}/project
      $ cd ${HOME}/project
      $ git clone https://github.com/jkjung-avt/jetson_nano.git
      $ cd jetson_nano
      $ ./install_basics.sh
      $ source ${HOME}/.bashrc
 
5\. INSTALL MOSQUITTO AND EDIT mosquitto.conf

      $ sudo apt install mosquitto mosquitto_clients
      
   *  add the following lines to /etc/mosquitto/mosquitto.conf

            $ listener 1883   
            $ allow_anonymous true
      
6\. INSTALL PAHO-MQTT
      
      $ pip3 install paho-mqtt
      $ pip install paho-mqtt
      
7\. CLONE THIS REPOSITORY 

      $ git clone git@github.com:GabeRibeiro/RSA_PROJETO_TP2G5.git
      
# HOW TO DEPLOY 
FOLLOW https://code.nap.av.it.pt/atcll/object-detection-camera/-/blob/jetson_2gb/yolov4/README.md

   **IMPORTANT: Each Jetson Nano has a broker hosted by itself. To make the messages go to the broker you need to change the line 101 in file utils/visualization.py to your Jetson Nano's own wireless interface IP and line 117 to the topic you would like to publish to.**

* Example : 

   *  **line 101** : mqttc.connect("192.168.66.22",1883,60) , with "192.168.66.22" being my wlan0 IP

   *  **line 117** : mqttc.publish('jetson2/object', payload=finallist, qos=0, retain=False) , with "jetson2/object" being the topic I chose


# HOW TO RUN (EXAMPLES)
On one Jetson Nano do

    $ mosquitto_sub -h "ip_of_other_jetson" -t "topic_chosen_in_other_jetson"
One the other Jetson Nano do    
      1\. USB CAMERA

          $ cd yolov4
          $ python3 trt_yolo.py --usb 0 -m yolov4-416 --width 1280 --height 720

      2\. VIDEO

          $ cd yolov4
          $ python3 trt_yolo.py --video ./vid2.mov -m yolov4-416

      3\. IMAGE

          $ cd yolov4
          $ python3 trt_yolo.py --image ./dog.jpg -m yolov4-416
