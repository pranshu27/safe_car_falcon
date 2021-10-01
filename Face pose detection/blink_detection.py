#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:43:31 2021

@author: pranshu
"""

import argparse
import time

import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import FileVideoStream, VideoStream
from scipy.spatial import distance as dist
import os
import sys
import time

os.chdir(r'/home/pranshu/Documents/GitHub/safe_car_falcon/Face pose detection')


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

PREDICTOR_PATH = os.path.join("models", "shape_predictor_68_face_landmarks.dat")

if not os.path.isfile(PREDICTOR_PATH):
    print("[ERROR] USE models/downloader.sh to download the predictor")
    sys.exit()


EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3


COUNTER = 0
TOTAL = 0


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


print("[INFO] starting video stream thread...")
vs = FileVideoStream('data/blink.mp4').start()
fileStream = True
#vs = VideoStream(src=0).start()
#fileStream = False
time.sleep(1.0)

flag = 0
start = 0


tmp_lst = []
while True:

    try:
        if fileStream and not vs.more():
            break
    
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        rects = detector(gray, 0)
    
        for rect in rects:
    
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
    
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
    
            ear = (leftEAR + rightEAR) / 2.0
    
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            
            
            if ear < EYE_AR_THRESH:
                print('close')
                
                if flag==0:
                    start = time.time()
                flag = 1
                
            else:
                
                if flag==1:
                    
                    print('open after close')
                    end = time.time() 
                    elapsed = end - start
                    tmp_lst.append(elapsed)
            
                    cv2.putText(
                        frame,
                        "Eyes closed for time: {:.2f}".format(elapsed),
                        (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                else:
                    print('open')
                start = 0 
                end = 0
                flag = 0

                
    
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        if key == ord("q"):
            
            break
    except:
        pass

cv2.destroyAllWindows()
vs.stop()

print(tmp_lst)


#time.time()