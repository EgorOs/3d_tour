#!/usr/bin/env python

import cv2
import numpy as np

class ViewData:
    
    ''' Object that stores image itself and additional information
        such as keypoints, descriptors, odometry '''

    def __init__(self, img):
        self.img = img
        self.linked_views = {}

    def getFeaturesData(self):
        features = cv2.xfeatures2d.SURF_create(hessianThreshold = 100, nOctaves = 4, nOctaveLayers = 3, extended = False, upright = False )
        RATIO_TEST_PARAMETER = 0.3
        self.kp, self.des = features.detectAndCompute(self.img,None)

    #descriptor_type = None

img_1 = cv2.imread('data/1.jpg')
img_2 = cv2.imread('data/2.jpg')

view_1 = ViewData(img_1)
view_2 = ViewData(img_2)
view_1.getFeaturesData()
view_2.getFeaturesData()
print([view.kp[i].pt for i in range(len(view.kp))])

# des = is like 10k*64 of float32 => 32bit*640k = 80kbytes # for desciptor only