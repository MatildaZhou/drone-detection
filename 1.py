import cv2
import numpy as np
import threading
import time


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

while True:
    if cam_cleaner.last_frame is not None:
        img = cam_cleaner.last_frame
   

        try:
            original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        except:
            break


        show=True
        if show:
            cv2.imshow('output', img)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

#vid.release()
#cv2.destroyAllWindows()
