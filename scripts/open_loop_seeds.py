"""
This is the open loop which uses a flat tissue and seeds on it (I'll probably do 8).

Make sure I exit early if all eight seeds are not detected correctly!!

Save the output! It might be useful, e.g.:

    clear; clear; python scripts/open_loop_seeds.py | tee images/seeds_v00/im_001.txt

where `im_001.txt` is assuming that we are on the 1st image. Be careful.
PS: This WILL override any existing `im_001.txt` file if we have it.
"""

from autolab.data_collector import DataCollector
from dvrk.robot import *
import cv2
import image_geometry
import numpy as np
import os
import pickle
import sys
import tfx
import utilities
np.set_printoptions(suppress=True)

###########################
# ADJUST THESE AS NEEDED! #
###########################

VERSION_INPUT  = '00'
VERSION_OUTPUT = '04' # See README in `images/` for details on which number to use.
ESC_KEYS = [27, 1048603]

# Adjust carefully!!
ROBOT_SPEED    = 0.06 # careful :-) 0.03 for regular, 0.06 for fast.
CLOSE_ANGLE    = 25 # I think 25 is good for pumpkin, 10 for sunflower.
TOPK_CONTOURS  = 8

# Loading stuff.
RF_REGRESSOR = pickle.load(open('config/mapping_results/random_forest_predictor_v'+VERSION_INPUT+'.p', 'r'))
PARAMETERS   = pickle.load(open('config/mapping_results/params_matrices_v'+VERSION_INPUT+'.p', 'r'))

# Rotations
ROTATION     = utilities.get_average_rotation(VERSION_INPUT)
TFX_ROTATION = tfx.tb_angles(ROTATION[0], ROTATION[1], ROTATION[2])

# Saving stuff
IMAGE_DIR = 'images/seeds_v'+VERSION_OUTPUT

# Offsets, some heuristic (e.g. x-coord), some (e.g. the z-coord) for safety.
ARM1_XOFFSET = -0.0005
ARM1_YOFFSET = 0.000
ARM1_ZOFFSET = -0.0015
ZOFFSET_SAFETY = 0.003 # What I actually use in practice

##########################
# END OF `CONFIGURATION` #
##########################


def motion_planning(contours_by_size, img, arm):
    """ The open loop.
    
    Parameters
    ----------
    contours_by_size: [list]
        A list of contours, arranged from largest area to smallest area.
    img: [np.array]
        Image the camera sees, in BGR form (not RGB).
    arm: [dvrk arm]
        Represents the arm we're using for the DVRK.
    """
    print("Identified {} contours but will keep top {}.".format(len(contours_by_size), TOPK_CONTOURS))
    img_for_drawing = img.copy()
    contours = list(contours_by_size)
    cv2.drawContours(img_for_drawing, contours, -1, (0,255,0), 3)
    places_to_visit = []

    # Iterate and find centers. We'll make the robot move to these centers in a sequence.
    # Note that duplicate contours should be detected beforehand.
    for i,cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        places_to_visit.append((cX,cY))

    # Collect only top K places to visit and insert ordering preferences. I do right to left.
    places_to_visit = places_to_visit[:TOPK_CONTOURS]
    places_to_visit = sorted(places_to_visit, key=lambda x:x[0], reverse=True)

    # Number the places to visit in an image so I see them.
    for i,(cX,cY) in enumerate(places_to_visit):
        cv2.putText(img=img_for_drawing, text=str(i), org=(cX,cY), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(0,0,0), thickness=2)
        cv2.circle(img=img_for_drawing, center=(cX,cY), radius=3, color=(0,0,255), thickness=-1)
        print(i,cX,cY)

    # Show image with contours + exact centers. Exit if it's not looking good. 
    # If good, SAVE IT!!! This protects me in case of poor perception causing problems.
    trial = len([fname for fname in os.listdir(IMAGE_DIR) if '.png' in fname])
    cv2.imshow("Top-K contours (exit if not looking good), trial index is {}".format(trial), img_for_drawing)
    utilities.call_wait_key()
    cv2.imwrite(IMAGE_DIR+"/im_"+str(trial).zfill(3)+".png", img_for_drawing)
    cv2.destroyAllWindows()

    # Given points in `places_to_visit`, must figure out the robot points for each.
    robot_points_to_visit = []
    print("\nHere's where we'll be visiting (left_pixels, robot_point):")
    for left_pixels in places_to_visit:
        robot_points_to_visit.append(
                utilities.left_pixel_to_robot_prediction(
                        left_pt=left_pixels,
                        params=PARAMETERS,
                        better_rf=RF_REGRESSOR,
                        ARM1_XOFFSET=ARM1_XOFFSET,
                        ARM1_YOFFSET=ARM1_YOFFSET,
                        ARM1_ZOFFSET=ARM1_ZOFFSET
                )
        )
        print("({}, {})\n".format(left_pixels, robot_points_to_visit[-1]))


    # With these robot points, now tell the robot to move to these points.
    # Here is where I apply the vertical offsets, FYI.
    arm.open_gripper(degree=90, time_sleep=2)
    for robot_pt in robot_points_to_visit:
        pos = [robot_pt[0], robot_pt[1], robot_pt[2]+ZOFFSET_SAFETY]
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, TFX_ROTATION), ROBOT_SPEED)

        pos[2] -= ZOFFSET_SAFETY
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, TFX_ROTATION), ROBOT_SPEED)
        arm.open_gripper(degree=CLOSE_ANGLE, time_sleep=2) # Need >=2 seconds!

        pos[2] += ZOFFSET_SAFETY
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, TFX_ROTATION), ROBOT_SPEED)

        arm.home(open_gripper=False, speed=ROBOT_SPEED)
        arm.open_gripper(degree=90, time_sleep=2) # Need >=2 seconds!


if __name__ == "__main__":
    """ See the top of the file for program-wide arguments. """
    arm, _, d = utilities.initializeRobots(sleep_time=2)
    arm.home()
    arm.close_gripper()
    print("arm home: {}".format(arm.get_current_cartesian_position()))
    utilities.show_images(d)
    motion_planning(d.left_contours_by_size, d.left_image, arm)
