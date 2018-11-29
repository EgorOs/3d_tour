#!/usr/bin/env python

import cv2
import numpy as np
import os

cap = cv2.VideoCapture('examples/test.mp4')

# Check if video opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

frame_count = 0
step = 120
while(cap.isOpened()):  

    ret, frame = cap.read()
    if ret:
        if frame_count % step == 0:
            cv2.imwrite('data/{}.jpg'.format(frame_count/step), frame)
        frame_count += 1
    #if no there is no new frames
    else:
        break
        
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()