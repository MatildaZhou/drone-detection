#================================================================
#
#   File name   : detection_custom.py
#   Author      : PyLessons
#   Created date: 2020-09-17
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3
#   Description : object detection image and video example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
#from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image, detect_realtime, detect_video, Load_Yolo_model, detect_video_realtime_mp,load_yolo_weights
from yolov3.utils import detect_video, Load_Yolo_model
from yolov3.configs import *

image_path   = "./IMAGES/drone1.png"

#video_path   = "rtsp://192.168.123.91/axis-media/media.amp"
#video_path = "./IMAGES/20210325_164706_2550.mp4"
video_path = "./IMAGES/cla.mp4"

#yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
#yolo.load_weights("./checkpoints/yolov3_custom") # use keras weights， i add
yolo = Load_Yolo_model()
#detect_image(yolo, image_path, "./IMAGES/drone1.png", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES,score_threshold=0.4, rectangle_colors=(255,0,0))
detect_video(yolo, video_path, "./IMAGES/output1.mp4", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, score_threshold=0.6,rectangle_colors=(255,0,0))
#detect_realtime(yolo, '', input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))

#detect_video_realtime_mp(video_path, "", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0), realtime=False)
#try1(yolo)
