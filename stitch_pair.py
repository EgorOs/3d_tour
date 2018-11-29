#!/usr/bin/env python

import numpy as np
import cv2

from matplotlib import pyplot as plt

# to explore directory
from os import listdir

descriptor = cv2.xfeatures2d.SURF_create(hessianThreshold=100, nOctaves=4,
                                         nOctaveLayers=3, extended=False,
                                         upright=False)
RATIO_TEST_PARAMETER = 0.3


def load_imgs_by_idx(dir_name):
    img_list = [cv2.imread('{}/{}.jpg'.format(dir_name, i)) for i in range(len(listdir(dir_name)))]
    img_list = [cv2.equalizeHist(i) for i in img_list]
    return img_list


img_list = load_imgs_by_idx('data')
img1, img2 = img_list[5], img_list[6]

kp1, des1 = descriptor.detectAndCompute(img1, None)
kp2, des2 = descriptor.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)
good = []
for m, n in matches:
    if m.distance < RATIO_TEST_PARAMETER * n.distance:
        good.append([m])  # [m] is required for drawMatches, for some reason it doesn't work with just m

img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

# Put warning in case if there is not enough matches, possibly increase RATIO_TEST_PARAMETER in that case

print('matches: {}, good_matches: {}'.format(len(matches), len(good)))

# plt.imshow(img3,),plt.show()


# Find homography https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
# https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
# if len(good) > threshold:


# Extract location of good matches
pts1 = np.zeros((len(good), 2), dtype=np.float32)
pts2 = np.zeros((len(good), 2), dtype=np.float32)

for i, match in enumerate(good):
    pts1[i, :] = kp1[match[0].queryIdx].pt
    pts2[i, :] = kp2[match[0].trainIdx].pt

# Find homography
h, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC)
print(h)

# Use homography
height, width, channels = img2.shape
img1Reg = cv2.warpPerspective(img1, h, (width + 800, height + 200))

# https://stackoverflow.com/questions/42167947/opencv-python-stitch-unordered-images
img1Reg[0:img2.shape[0], 0:img2.shape[1]] = img2
plt.imshow(img1Reg, ), plt.show()
cv2.imwrite('result.png', img1Reg)
