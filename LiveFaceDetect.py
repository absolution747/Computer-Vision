# -*- coding: utf-8 -*-
"""
to run this code go to cmd prompt, change directory to the location of the code
file, and the run the following command

python LiveFaceDetect.py -p deploy.prototxt.txt -m res10_300x300_ssd_iter_140000.caffemodel -c 0.6

@author: Karan Shetty
"""

from imutils.video import VideoStream
import numpy as np
import cv2
import argparse
import imutils
import time

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required = True, 
                help ="path to caffe 'deploy'prototxt file")
ap.add_argument("-m", "--model", required = True, 
                help = "path to caffe pretrained model")
ap.add_argument("-c", "--confidence", type = float, default = 0.5,
                help = "minimum confidence probability threshold")
args = vars(ap.parse_args())

print("[INFO] loading model....")
net = cv2.dnn.readNetFromCaffe(args["prototxt"],args["model"])

print("[INFO] starting video stream....")
vs = VideoStream(src= 0).start()
time.sleep(2.0)


while True:
    frame = vs.read()
    frame = imutils.resize(frame, width = 400)
    (h,w) = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,
                             (300,300), (104.0,177.0,123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0,detections.shape[2]):
        
        confidence = detections[0,0,i,2]
        
        if confidence < args["confidence"]:
            continue
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        (StartX, StartY, EndX, EndY) = box.astype("int")
        
        text = "{:.2f}%".format(confidence*100)
        y = StartY -10 if StartY > 20 else StartY + 10
        
        cv2.rectangle(frame, (StartX,StartY),(EndX,EndY),(0,0,255),2)
        cv2.putText(frame,text,(StartX,y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, 
                    (0,0,255), 2)
       
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
 
cv2.destroyAllWindows()
vs.stop()
    