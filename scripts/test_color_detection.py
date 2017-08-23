""" 
Test HSV saturation or other color detection stuff. I'm not optimistic
about it but I might as well try, and it gives me protection in case I
have the wrong colors.
"""

import cv2
import image_geometry
import numpy as np
import os
import pickle
import sys
import utilities
from autolab.data_collector import DataCollector
from dvrk.robot import *
np.set_printoptions(suppress=True)

# DOUBLE CHECK, and see `images/README.md` for details.
OUTVERSION   = '99' # for _storing_ stuff, use 99 for debugging
VERSION      = '00' # for _loading_ stuff

OUTPUT_FILE  = 'config/calibration_results/data_v'+OUTVERSION+'.p'
IMDIR        = 'images/check_regressors_v'+OUTVERSION+'/'
ESC_KEYS     = [27, 1048603]
USE_RF       = True
MAX_NUM_ADD  = 36
ROTATION     = utilities.get_average_rotation(VERSION)

##########################
# END OF `CONFIGURATION` #
##########################

if __name__ == "__main__":
    arm, _, d = utilities.initializeRobots()
    print("current arm position: {}".format(arm.get_current_cartesian_position()))
    arm.close_gripper()
    print("current arm position: {}".format(arm.get_current_cartesian_position()))
    params = pickle.load(open('config/mapping_results/params_matrices_v'+VERSION+'.p', 'r'))
    better_rf = pickle.load(open('config/mapping_results/random_forest_predictor_v'+VERSION+'.p', 'r'))

    # Actual experimentation ...
    utilities.call_wait_key(cv2.imshow("left image", d.left_image))

    frame = d.left_image.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([110,100,100])
    upper = np.array([130,255,255])
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    utilities.call_wait_key(cv2.imshow("left image", res))
