"""
The purpose of this script is for me to check the camera location for the dvrk.
Sometimes I accidentally move the camera, so this will serve to stop that from
having too damaging effects by saving where the bounding box is stored. I also
save the contour locations but this will be harder to check. Oh well.
"""

from dvrk.robot import *
from autolab.data_collector import DataCollector
import cv2
import numpy as np
import pickle
import sys
import time
import utilities
DIR = "camera_location/"

VERSION = '10'

# Don't change these!!
left_xx = 650
left_yy = 100
left_ww = 775
left_hh = 700

right_xx = 575
right_yy = 100
right_ww = 775
right_hh = 700


# todo maybe put some calibration method here to help fix it as needed?
# if i make a mistake?
def fix_calibration():
    pass


def save_bounding_box(image, left):
    xx = left_xx if left else right_xx
    yy = left_yy if left else right_yy
    ww = left_ww if left else right_ww
    hh = left_hh if left else right_hh
    name = 'left' if left else 'right'

    cv2.imwrite(DIR+'calibration_blank_image_'+name+'_v'+VERSION+'.jpg', image)
    cv2.rectangle(image, (xx,yy), (xx+ww, yy+hh), (0,255,0), 2)

    cv2.putText(img=image, text='{},{}'.format(xx,yy),       org=(xx,yy),       
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
    cv2.putText(img=image, text='{},{}'.format(xx,yy+hh),    org=(xx,yy+hh),    
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
    cv2.putText(img=image, text='{},{}'.format(xx+ww,yy),    org=(xx+ww,yy),    
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)
    cv2.putText(img=image, text='{},{}'.format(xx+ww,yy+hh), org=(xx+ww,yy+hh), 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=2)

    cv2.imshow("Camera Image w/BBox", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(DIR+'calibration_bbox_image_'+name+'_v'+VERSION+'.jpg', image)


if __name__ == "__main__":
    arm1, _, d = utilities.initializeRobots()
    arm1.close_gripper()

    image = d.left_image.copy()
    save_bounding_box(image, left=True)
    pickle.dump(d.left_contours, open(DIR+'contours_left_v'+VERSION+'.p', 'w'))

    image = d.right_image.copy()
    save_bounding_box(image, left=False)
    pickle.dump(d.right_contours, open(DIR+'contours_right_v'+VERSION+'.p', 'w'))
