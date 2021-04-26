import threading
import cv2
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
#from yolov3.yolov4 import Create_Yolo
from yolov3.utils import *
from yolov3.configs import *

Yolo = Load_Yolo_model()
# Define the thread that will continuously pull frames from the camera
class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()
    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()
# Start the camera
camera = cv2.VideoCapture("rtsp://192.168.123.91/axis-media/media.amp")
# Start the cleaning thread
cam_cleaner = CameraBufferCleanerThread(camera)
count = 0
times, times_2 = [], []
while True:
    if cam_cleaner.last_frame is not None:
        img = cam_cleaner.last_frame


        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break

        if 0==0:
           
            image_data = image_preprocess(np.copy(original_image), [416, 416])
            image_data = image_data[np.newaxis, ...].astype(np.float32)

            t1 = time.time()
            if YOLO_FRAMEWORK == "tf":
                pred_bbox = Yolo.predict(image_data)
            elif YOLO_FRAMEWORK == "trt":
                batched_input = tf.constant(image_data)
                result = Yolo(batched_input)
                pred_bbox = []
                for key, value in result.items():
                    value = value.numpy()
                    pred_bbox.append(value)
        
            t2 = time.time()
        
            pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
            pred_bbox = tf.concat(pred_bbox, axis=0)

            bboxes = postprocess_boxes(pred_bbox, original_image, 416, 0.4)
            bboxes = nms(bboxes, 0.45, method='nms')
        
            image = draw_bbox(original_image, bboxes, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))

            t3 = time.time()
            times.append(t2-t1)
            times_2.append(t3-t1)
        
            times = times[-20:]
            times_2 = times_2[-20:]

            ms = sum(times)/len(times)*1000
            fps = 1000 / ms
            fps2 = 1000 / (sum(times_2)/len(times_2)*1000)
        
            image = cv2.putText(image, "Time: {:.1f}FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        #CreateXMLfile("XML_Detections", str(int(time.time())), original_image, bboxes, read_class_names(CLASSES))
        
            print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
           
            show=True
            if show:
                cv2.imshow('output', image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
        
        
    
        #cv2.imwrite("frame%d.jpg" % count, cam_cleaner.last_frame)
        #count += 1
        #print("Image saved")
    #else:
        #print("No camera open")






