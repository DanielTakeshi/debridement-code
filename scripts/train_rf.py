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
import utilities as utils
from autolab.data_collector import DataCollector
from dvrk.robot import *
np.set_printoptions(suppress=True)

# Double check these as needed. Here, the version numbers don't change that much.
# Actually, the version numbers should be the same here. Just double check please.
OUTVERSION   = '01' # for _storing_ stuff
VERSION      = '01' # for _loading_ stuff, changes with tool (+ yaw angle...)!!

# Other stuff.
OUTPUT_FILE  = 'config/calibration_results/data_for_rf_v'+OUTVERSION+'.p'
ROTATION     = utils.get_rotation_from_version(VERSION)
MAX_NUM_ADD  = 36


if __name__ == "__main__":
    print("Rotation: {}".format(ROTATION))
    arm, _, d = utils.initializeRobots()
    print("current arm position: {}".format(arm.get_current_cartesian_position()))
    utils.move(arm, utils.HOME_POS, ROTATION, 'Slow')
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
        if utils.filter_point(cX,cY, 500,1500,50,1000):
            continue
        if num_added == MAX_NUM_ADD:
            break

        image = image_original.copy()
        cv2.circle(img=image, center=(cX,cY), radius=50, color=(0,0,255), thickness=1)
        cv2.circle(img=image, center=(cX,cY), radius=4, color=(0,0,255), thickness=-1)
        cv2.drawContours(image=image, contours=[approx], contourIdx=-1, color=(0,255,0), thickness=3)
        cv2.imshow("Press ESC if this isn't a desired contour (or a duplicate), press any other key to proceed w/robot movement. Note that num_added = {}".format(num_added), image)
        firstkey = cv2.waitKey(0) 

        if firstkey not in utils.ESC_KEYS:
            # First, determine where the robot will move to based on the pixels. This forms the
            # input to the good random forest. Normally we'd have the good RF inside there as input
            # as `better_rf` but of course we do not have that yet! This is the rigid body only.
            target = utils.left_pixel_to_robot_prediction(
                    left_pt=(cX,cY), 
                    params=params, 
                    better_rf=None,
                    ARM1_XOFFSET=0,
                    ARM1_YOFFSET=0,
                    ARM1_ZOFFSET=0,
                    USE_RF=False
            )

            # Robot moves to that point. Will likely be off just a bit. 
            pos = [target[0], target[1], target[2]]
            utils.move(arm, pos, ROTATION, 'Slow')
            time.sleep(5)

            # This should NOT be the input to the RF! We should use `pos` or `target` instead.
            # BUT we still need it to get correct offsets in the _workspace_ (i.e. real world)!!
            predicted_pos = arm.get_current_cartesian_position() 

            # IMPORTANT! Now I do some human motion!!
            updated_image_copy = (d.left_image).copy()
            window_name = "Tried to move to ({},{}), position {}. ADJUST IT AS NEEDED then press any key other than ESC! Or ESC if I made a mistake and need to skip this".format(cX,cY,predicted_pos)
            cv2.imshow(window_name, updated_image_copy)
            key = cv2.waitKey(0)
            if key in utils.ESC_KEYS:
                continue
            cv2.destroyAllWindows()

            # Now record the new position of the arm, and save it.
            new_pos = arm.get_current_cartesian_position()
            data_pt = {'original_robot_point_prediction':          pos,
                       'measured_robot_point_before_human_change': predicted_pos, 
                       'measured_robot_point_after_human_change':  new_pos}
            utils.storeData(OUTPUT_FILE, data_pt)

            # Some stats for debugging, etc.
            num_added += 1
            print("contour {}, data_pt: {} (# added: {})".format(i, data_pt, num_added))
        else:
            print("(not storing contour {})".format(i))

        utils.move(arm, utils.HOME_POS, ROTATION, 'Slow')
        arm.close_gripper()
        cv2.destroyAllWindows()
