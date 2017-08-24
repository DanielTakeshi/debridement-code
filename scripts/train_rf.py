""" 
AFTER we train the rigid body, this will go through the points and apply the predictor.
I'll get systematic errors (at least I should) so I will MANUALLY adjust the robot motion. 
Then train a random forest predictor on that. THEN do again the click-and-crop thingy.
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

# Double check these as needed. CHECK THE NUMBERS, i.e. v00, v01, etc.
OUTVERSION   = '10' # for _storing_ stuff, use 99 for debugging
VERSION      = '10' # for _loading_ stuff

# Other stuff, e.g. consider z-offset if the paper is at risk of being damaged.
OUTPUT_FILE  = 'config/calibration_results/data_for_rf_v'+OUTVERSION+'.p'
ESC_KEYS     = [27, 1048603]
C_LEFT_INFO  = pickle.load(open('config/camera_info_matrices/left.p',  'r'))
C_RIGHT_INFO = pickle.load(open('config/camera_info_matrices/right.p', 'r'))
ROTATION     = utilities.get_average_rotation(VERSION)
MAX_NUM_ADD  = 36


if __name__ == "__main__":
    arm, _, d = utilities.initializeRobots()
    print("current arm position: {}".format(arm.get_current_cartesian_position()))
    arm.home()
    arm.close_gripper()
    print("current arm position: {}".format(arm.get_current_cartesian_position()))

    # I will be IGNORING the random forest here, because it doesn't really work.
    params = pickle.load(open('config/mapping_results/params_matrices_v'+VERSION+'.p', 'r'))

    # Use the d.left_image for calibration.
    cv2.imwrite("images/left_image.jpg", d.left_image)
    image_original = cv2.imread("images/left_image.jpg")
    num_added = 0

    # Iterate through valid contours from the _left_ camera (we'll simulate right camera).
    for i, (cX, cY, approx, peri) in enumerate(d.left_contours):  
        if utilities.filter_point(cX,cY, 500,1500,50,1000):
            continue
        if num_added == MAX_NUM_ADD:
            break

        image = image_original.copy()
        cv2.circle(img=image, center=(cX,cY), radius=50, color=(0,0,255), thickness=1)
        cv2.circle(img=image, center=(cX,cY), radius=4, color=(0,0,255), thickness=-1)
        cv2.drawContours(image=image, contours=[approx], contourIdx=-1, color=(0,255,0), thickness=3)
        cv2.imshow("Press ESC if this isn't a desired contour (or a duplicate), press any other key to proceed w/robot movement. Note that num_added = {}".format(num_added), image)
        firstkey = cv2.waitKey(0) 

        if firstkey not in ESC_KEYS:
            # First, determine where the robot will move to based on the pixels. This forms the
            # input to the good random forest.
            target = utilities.left_pixel_to_robot_prediction(
                    left_pt=(cX,cY), 
                    params=params, 
                    better_rf=None,
                    ARM1_XOFFSET=0,
                    ARM1_YOFFSET=0,
                    ARM1_ZOFFSET=0,
                    USE_RF=False
            )
            pos = [target[0], target[1], target[2]]
            rot = tfx.tb_angles(ROTATION[0], ROTATION[1], ROTATION[2])

            # Robot moves to that point and THEN lowers itself. Will likely be off. 
            # I think 6 seconds is enough for the camera to refresh. But save the frame!
            arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)

            # This should NOT be the input to the RF! We should use `pos` or `target` instead.
            predicted_pos = arm.get_current_cartesian_position() 

            # IMPORTANT! Now I do some human motion!!
            updated_image_copy = (d.left_image).copy()
            window_name = "Tried to move to ({},{}), position {}. ADJUST IT AS NEEDED then press any key other than ESC! Or ESC if I made a mistake and need to skip this".format(cX,cY,predicted_pos)
            cv2.imshow(window_name, updated_image_copy)
            key = cv2.waitKey(0)
            if key in ESC_KEYS:
                continue
            cv2.destroyAllWindows()

            # Now record the new position of the arm, and save it.
            new_pos = arm.get_current_cartesian_position()
            data_pt = {'predicted_pos': pos, 'new_pos': new_pos}
            utilities.storeData(OUTPUT_FILE, data_pt)

            # Some stats for debugging, etc.
            num_added += 1
            print("contour {}, data_pt: {} (# added: {})".format(i, data_pt, num_added))
        else:
            print("(not storing contour {})".format(i))

        arm.home()
        arm.close_gripper()
        cv2.destroyAllWindows()
