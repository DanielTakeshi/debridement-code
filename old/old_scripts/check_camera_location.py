"""
The purpose of this script is for me to check the camera location for the dvrk.
Sometimes I accidentally move the camera, so this will serve to stop that from
having too damaging effects by saving where the bounding box is stored.
"""

import environ
from dvrk.robot import *
from autolab.data_collector import DataCollector
import cv2
import numpy as np
import pickle
import sys
IMDIR = "images/check_calibration/"


# Don't change these!!
xx = 450
yy = 200

ww = 775
hh = 700

def initializeRobots():
    d = DataCollector()
    r1 = robot("PSM1") # left (but my right)
    r2 = robot("PSM2") # right (but my left)
    time.sleep(2)
    return (r1,r2,d)


# todo maybe put some calibration method here to help fix it as needed?
# if i make a mistake?
def fix_calibration():
    pass


def save_bounding_box(image):
    cv2.imwrite(IMDIR+'calibration_blank_image.jpg', image)
    cv2.rectangle(image, (xx,yy), (xx+ww, yy+hh), (0,255,0), 2)

    cv2.putText(img=image, text='{},{}'.format(xx,yy),       org=(xx,yy),       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
    cv2.putText(img=image, text='{},{}'.format(xx,yy+hh),    org=(xx,yy+hh),    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
    cv2.putText(img=image, text='{},{}'.format(xx+ww,yy),    org=(xx+ww,yy),    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
    cv2.putText(img=image, text='{},{}'.format(xx+ww,yy+hh), org=(xx+ww,yy+hh), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)

    cv2.imshow("Left Camera Image", image)
    cv2.waitKey(0)
    cv2.imwrite(IMDIR+'calibration_bbox_image.jpg', image)


if __name__ == "__main__":
    arm1, _, d = initializeRobots()
    arm1.close_gripper()
    cv2.imshow("Left Camera Image", d.left_image)
    cv2.waitKey(0)
    image = d.left_image.copy()
    save_bounding_box(image)
