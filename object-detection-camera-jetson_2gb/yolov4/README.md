# NAP YOLOv4 Object Detection Algorithm

Author: Gonçalo Perna, Pedro Teixeira

Last update: June 12, 2022 

Sources: [jkjung-avt/tensorrt_demos: TensorRT MODNet, YOLOv4, YOLOv3, SSD, MTCNN, and GoogLeNet (github.com)](https://github.com/jkjung-avt/tensorrt_demos) 

## Requirements

*   NVIDIA Jetson Nano
*   TensorRT 8.0.x+.

You can check which version of TensorRT has been installed on your Jetson system by looking at file names of the libraries.   
For example, TensorRT v5.1.6 (JetPack-4.2.2) was present on one of my Jetson Nano DevKits.

    $ ls /usr/lib/aarch64-linux-gnu/libnvinfer.so*
    /usr/lib/aarch64-linux-gnu/libnvinfer.so
    /usr/lib/aarch64-linux-gnu/libnvinfer.so.5
    /usr/lib/aarch64-linux-gnu/libnvinfer.so.5.1.6

*   "cv2" (OpenCV) module for python3\. You can use the "cv2" module which came in the JetPack.   
    Or, if you'd prefer building your own, refer to [Installing OpenCV 3.4.6 on Jetson Nano](https://jkjung-avt.github.io/opencv-on-nano/) for how to build from source and install opencv-3.4.6 on your Jetson system.

## How to deploy

1\. Clone repository [atcll / object-detection-camera · GitLab](https://code.nap.av.it.pt/atcll/object-detection-camera)

2\. How to setup:

<figure class="table">

<table>

<tbody>

<tr>

<td>
A. Install Dependencies. 

    $ cd yolov4
    $ chmod +x install_dependencies.sh
    $ chmod +x start_detection.sh
    $ ./install_dependencies.sh

The script ("install_dependencies.sh") can take a while (more than one hour) 

B. Go to the "plugins/" subdirectory and build the "yolo_layer" plugin. When done, a "libyolo_layer.so" would be generated.

    $ cd yolov4/plugins
    $ make

C. Download the pre-trained yolov3/yolov4 COCO models and convert the targeted model to ONNX and then to TensorRT engine. I use "yolov4-416" as example below. (Supported models: "yolov3-tiny-288", "yolov3-tiny-416", "yolov3-288", "yolov3-416", "yolov3-608", "yolov3-spp-288", "yolov3-spp-416", "yolov3-spp-608", "yolov4-tiny-288", "yolov4-tiny-416", "yolov4-288", "yolov4-416", "yolov4-608", "yolov4-csp-256", "yolov4-csp-512", "yolov4x-mish-320", "yolov4x-mish-640", and [custom models](https://jkjung-avt.github.io/trt-yolov3-custom/) such as "yolov4-416x256".)

    $ cd yolov4/yolo
    $ ./download_yolo.sh
    $ python3 yolo_to_onnx.py -m yolov4-416
    $ python3 onnx_to_tensorrt.py -m yolov4-416

Lines ("python3 yolo_to_onnx.py -m model") and ("python3 onnx_to_tensorrt.py -m model") need to be repeated for every model that is tested

The last step ("onnx_to_tensorrt.py") takes a little bit more than half an hour to complete on my Jetson Nano DevKit. When that is done, the optimized TensorRT engine would be saved as "yolov4-416.trt".

In case "onnx_to_tensorrt.py" fails (process "Killed" by Linux kernel), it could likely be that the Jetson platform runs out of memory during conversion of the TensorRT engine. This problem might be solved by adding a larger swap file to the system.

D. (Optional) Simple test with a downloaded image, in order to watch the output image, restart ssh connection with the flag -X (ssh user@address -X)

    $ cd yolov4
    $ wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg -O ./dog.jpg
    $ python3 trt_yolo.py --image ./dog.jpg -m yolov4-416

References: [jkjung-avt/tensorrt_demos: TensorRT MODNet, YOLOv4, YOLOv3, SSD, MTCNN, and GoogLeNet (github.com),](https://github.com/jkjung-avt/tensorrt_demos) [Process killed in onnx_to_tensorrt.py Demo#5](https://github.com/jkjung-avt/tensorrt_demos/issues/344).

</td>

</tr>

</tbody>

</table>

</figure>

3\. Alternatively, it is possible to copy the folder in `/home/jetson/object-camera-detection` available in `atcll-p1-jetson.nap.av.it.pt`, `atcll-p3-jetson.nap.av.it.pt` or `atcll-p22-jetson.nap.av.it.pt` to the new Jetson.

4\. Change the start script to use the information of the camera.

`# cd yolov4`

`nano camera.properties`

Change arguments as required:
*   `CAMERA_RTSP`: Change rtsp://username:password@atcll-p22-camera.nap.av.it.pt to the RTSP URI of the camera feed.
*   `CAMERA_RTSP_HEIGHT` and `CAMERA_RTSP_WIDTH` of the video stream: Change as needed.
*   `CAMERA_LATITUDE` and `CAMERA_LONGITUDE`: Location of the camera: Change as needed with the appropriated format, e.g. 40.63972 -8.64352
*   `CAMERA_ROTATION`: rotation of the camera feed: Change as needed. Possible values are 0 (no rotation), 90, 180 and 270.
    **IMPORTANT: If the camera model is X, you need to rotate the video stream before running the detection service to ensure detection will work properly.**
*   `DETECTION_ADD_X_COORDINATES` and `DETECTION_ADD_Y_COORDINATES`: to fine-tune the detection coordinates to an approximation of the real WGS84 coordinates.


5\. Configure detection algorithm as a system service so it runs at startup.

`# cd yolov4`

`sudo cp camera_detection.service /etc/systemd/system/camera_detection.service`

`systemctl start camera_detection.service`

`systemctl status camera_detection.service` to verify if the service is running.
