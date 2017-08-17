"""
Double check all my calibration points. Purely a debugging method.
"""

import cv2
import numpy as np
import pickle
import sys
import time


def show_points(image, left):
    name = 'config/calib_circlegrid_right_v00_ONELIST.p' 
    if left:
        name = 'config/calib_circlegrid_left_v00_ONELIST.p' 
    contours = pickle.load(open(name, 'r'))

    for i,(pos,rot,cx,cy) in enumerate(contours):
        cv2.putText(img=image, text='{},{}'.format(cx,cy), org=(cx,cy),       
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(0,0,0), thickness=2)
        cv2.putText(img=image, text='{}'.format(i), org=(cx-5,cy-5),    
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(255,0,0), thickness=2)

    cv2.imshow("Image w/points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_points(cv2.imread('camera_location/calibration_blank_image_left.jpg').copy(),  left=True)
    show_points(cv2.imread('camera_location/calibration_blank_image_right.jpg').copy(), left=False)
