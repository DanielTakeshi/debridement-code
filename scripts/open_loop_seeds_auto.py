"""
Open loop for the case when we have a bunch of seeds scattered with random rotations.
This is designed to use data originally from the automatically collected trajectories.

Usage example:

    python scripts/open_loop_seeds_auto.py

Notes:

- Make sure I exit early if all eight seeds are not detected correctly!!
- Images should be saved of the initial contours to check for potential errors.
- Don't use `tee` as I should be doing this inside code automatically.
- Add guards to prevent accidental overriding of files.
"""

import argparse
import cv2
import image_geometry
import numpy as np
import os
import pickle
import sys
import tfx
import utilities as utils
from autolab.data_collector import DataCollector
from dvrk.robot import *
from keras.models import load_model
np.set_printoptions(suppress=True)


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

    # Load the initial neural network. Don't forget the mean and standard deviation!!!
    f_network = load_model(PARAMS['modeldir'])
    net_mean  = PARAMS['X_mean']
    net_std   = PARAMS['X_std']

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

    # With robot points, tell it to _move_ to these points. Apply vertical offsets here, FYI.
    # Using the `linear_interpolation` is slower and jerkier than just moving directly.
    arm.open_gripper(degree=90, time_sleep=2)
    for robot_pt in robot_points_to_visit:
        pos = [robot_pt[0], robot_pt[1], robot_pt[2]+ZOFFSET_SAFETY]
        utilities.move(arm, pos, ROTATION, SPEED_CLASS)

        pos[2] -= ZOFFSET_SAFETY
        utilities.move(arm, pos, ROTATION, SPEED_CLASS)
        arm.open_gripper(degree=CLOSE_ANGLE, time_sleep=2) # Need >=2 seconds!

        pos[2] += ZOFFSET_SAFETY
        utilities.move(arm, pos, ROTATION, SPEED_CLASS)

        utilities.move(arm, HOME_POS, ROTATION, SPEED_CLASS)
        arm.open_gripper(degree=90, time_sleep=2) # Need >=2 seconds!


if __name__ == "__main__":
    """ 
    Set up a configuration here. See `images/README.md` and subdirectories there.
    Use 2X for neural net stuff. 
    """
    pp = argparse.ArgumentParser()
    pp.add_argument('--version_in', type=int, default=0, help='For now, it\'s 0')
    pp.add_argument('--version_out', type=int, help='See `images/README.md`, etc.')
    pp.add_argument('--x_offset', type=float, default=0.000)
    pp.add_argument('--y_offset', type=float, default=0.000)
    pp.add_argument('--z_offset', type=float, default=0.000)
    pp.add_argument('--max_num_add', type=int, default=35) # If I do 36, I'll be restarting a lot. :-)
    pp.add_argument('--guidelines_dir', type=str, default='traj_collector/guidelines.p')
    pp.add_argument('--use_rf_correctors', action='store_true')
    args = pp.parse_args()

    # Check the image versions, etc.
    IN_VERSION  = str(args.version_in).zfill(2)
    OUT_VERSION = str(args.version_out).zfill(2)
    IMAGE_DIR   = 'images/seeds_v'+OUT_VERSION+'/'
    assert (OUT_VERSION is not None) and (IN_VERSION is not None)

    # We want the image directory to exist; different trials will save in this directory.
    assert os.path.exists(IMAGE_DIR)
    trial = len([fname for fname in os.listdir(IMAGE_DIR) if '.png' in fname])
    trial_str = str(trial).zfill(3)
    tosave_img = IMAGE_DIR+"img_"+trial_str+".png"
    tosave_txt = IMAGE_DIR+"img_"+trial_str+".txt"
    print("Inside image dir {}, this code will save as trial {}".format(IMAGE_DIR, trial))
    print("Will save image as:       {}".format(tosave_img))
    print("Will save text output as: {}".format(tosave_txt))

    # Load the parameters ...
    PARAMS = {}
    yaws = [90, 45, 0, -45, -90, None]      # IMPORTANT!
    nums = [str(x) for x in range(20,25+1)] # IMPORTANT!
    assert len(yaws) == len(nums)
    head = 'config/mapping_results/rf_human_guided_yesz_v'
    for (yy,nn) in zip(yaws,nums):
        if yy is not None: # Ignore `None` case for now.
            PARAMS[yy] = pickle.load( open(head+str(nn)+'.p','r') )
    PARAMS['nnet'] = pickle.load(open('config/mapping_results/auto_params_matrices_v'+IN_VERSION+'.p','r'))
    print("Our parameters has keys: {}".format(PARAMS.keys()))

    # Now get the robot arm initialized and test it out!
    arm, _, d = utils.initializeRobots()
    utils.home(arm)
    print("current arm position: {}".format(arm.get_current_cartesian_position()))
    arm.close_gripper()

    # Let's do a pre-processing step where we first filter for contours and inspect.

    ## # NOW do the motion planning!
    ## motion_planning(d.left_image,
    ##                 d.right_image,
    ##                 left_contours,
    ##                 right_contours,
    ##                 left_angles,
    ##                 right_angles,
    ##                 tosave_img,
    ##                 tosave_txt,
    ##                 PARAMS, arm, d)
