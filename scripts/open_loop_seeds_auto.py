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
np.set_printoptions(suppress=True, linewidth=200)


def motion_planning(l_img, r_img, l_cnts, r_cnts, tosave_txt, PARAMS, arm, d):
    """ The open loop. It doesn't return anything, but it saves a lot.
    
    Parameters
    ----------
    l_img: [img]
        A copy of the left image, if I need it (but I actually don't).
    r_img: [img]
        A copy of the right image, if I need it (but I actually don't).
    l_cnts: [list]
        List of processed contours from the left image, each of the form (cX,cY,angle,yaw).
    r_cnts: [list]
        List of processed contours from the right image, each of the form (cX,cY,angle,yaw).
    tosave_txt: [string]
        Path to save the text file output.
    PARAMS: [dict]
        Dictionary of parameters to use!
    arm: [dvrk arm]
        Represents the arm we're using for the DVRK.
    d: [dvrk]
        The data collector.
    """
    # Load the initial neural network. Don't forget the mean and standard deviation!!!
    f_network = load_model(PARAMS['nnet']['modeldir'])
    net_mean  = PARAMS['nnet']['X_mean']
    net_std   = PARAMS['nnet']['X_std']
    print("network mean: {}".format(net_mean))
    print("network std:  {}".format(net_std))

    ## print("Identified {} contours but will keep top {}.".format(len(contours_by_size), TOPK_CONTOURS))
    ## img_for_drawing = img.copy()
    ## contours = list(contours_by_size)
    ## cv2.drawContours(img_for_drawing, contours, -1, (0,255,0), 3)
    ## places_to_visit = []

    ## # Iterate and find centers. We'll make the robot move to these centers in a sequence.
    ## # Note that duplicate contours should be detected beforehand.
    ## for i,cnt in enumerate(contours):
    ##     M = cv2.moments(cnt)
    ##     if M["m00"] == 0: continue
    ##     cX = int(M["m10"] / M["m00"])
    ##     cY = int(M["m01"] / M["m00"])
    ##     places_to_visit.append((cX,cY))

    ## # Collect only top K places to visit and insert ordering preferences. I do right to left.
    ## places_to_visit = places_to_visit[:TOPK_CONTOURS]
    ## places_to_visit = sorted(places_to_visit, key=lambda x:x[0], reverse=True)

    ## # Number the places to visit in an image so I see them.
    ## for i,(cX,cY) in enumerate(places_to_visit):
    ##     cv2.putText(img=img_for_drawing, text=str(i), org=(cX,cY), 
    ##                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
    ##                 fontScale=1, color=(0,0,0), thickness=2)
    ##     cv2.circle(img=img_for_drawing, center=(cX,cY), radius=3, color=(0,0,255), thickness=-1)
    ##     print(i,cX,cY)

    ## # Show image with contours + exact centers. Exit if it's not looking good. 
    ## # If good, SAVE IT!!! This protects me in case of poor perception causing problems.
    ## trial = len([fname for fname in os.listdir(IMAGE_DIR) if '.png' in fname])
    ## cv2.imshow("Top-K contours (exit if not looking good), trial index is {}".format(trial), img_for_drawing)
    ## utilities.call_wait_key()
    ## cv2.imwrite(IMAGE_DIR+"/im_"+str(trial).zfill(3)+".png", img_for_drawing)
    ## cv2.destroyAllWindows()

    ## # Given points in `places_to_visit`, must figure out the robot points for each.
    ## robot_points_to_visit = []
    ## print("\nHere's where we'll be visiting (left_pixels, robot_point):")
    ## for left_pixels in places_to_visit:
    ##     robot_points_to_visit.append(
    ##             utilities.left_pixel_to_robot_prediction(
    ##                     left_pt=left_pixels,
    ##                     params=PARAMETERS,
    ##                     better_rf=RF_REGRESSOR,
    ##                     ARM1_XOFFSET=ARM1_XOFFSET,
    ##                     ARM1_YOFFSET=ARM1_YOFFSET,
    ##                     ARM1_ZOFFSET=ARM1_ZOFFSET
    ##             )
    ##     )
    ##     print("({}, {})\n".format(left_pixels, robot_points_to_visit[-1]))

    ## # With robot points, tell it to _move_ to these points. Apply vertical offsets here, FYI.
    ## # Using the `linear_interpolation` is slower and jerkier than just moving directly.
    ## arm.open_gripper(degree=90, time_sleep=2)
    ## for robot_pt in robot_points_to_visit:
    ##     pos = [robot_pt[0], robot_pt[1], robot_pt[2]+ZOFFSET_SAFETY]
    ##     utilities.move(arm, pos, ROTATION, SPEED_CLASS)

    ##     pos[2] -= ZOFFSET_SAFETY
    ##     utilities.move(arm, pos, ROTATION, SPEED_CLASS)
    ##     arm.open_gripper(degree=CLOSE_ANGLE, time_sleep=2) # Need >=2 seconds!

    ##     pos[2] += ZOFFSET_SAFETY
    ##     utilities.move(arm, pos, ROTATION, SPEED_CLASS)

    ##     utilities.move(arm, HOME_POS, ROTATION, SPEED_CLASS)
    ##     arm.open_gripper(degree=90, time_sleep=2) # Need >=2 seconds!


def get_good_contours(proc_image, image, bb, savedir, max_num_add=None):
    """ 
    Adapted from `click_and_crop_v3.py`, except that we have to make the contours.
    Here, we're going to inspect and check that the contours are reasonable.
    Returns a list of processed contours that I'll then use for later.
    """
    cv2.imshow("Now detecting contours for this image.", proc_image)
    key = cv2.waitKey(0)
    if key in utils.ESC_KEYS:
        sys.exit()
    (cnts, _) = cv2.findContours(proc_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    processed = []

    for c in cnts:
        try:
            # Find the centroids of the contours in _pixel_space_. :)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            if utils.filter_point(cX,cY,xlower=bb[0],xupper=bb[0]+bb[2],ylower=bb[1],yupper=bb[1]+bb[3]):
                continue

            # Now fit an ellipse!
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(image, ellipse, (0,255,0), 2)
            name = "Is this ellipse good? ESC to skip it, else add it."
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, 2000, 4000)
            cv2.imshow(name, image)
            firstkey = cv2.waitKey(0) 

            if firstkey not in utils.ESC_KEYS:
                angle = ellipse[2]
                yaw = utils.opencv_ellipse_angle_to_robot_yaw(angle)
                processed.append( (cX,cY,angle,yaw) )
                cv2.putText(img=image,
                        text="{},{:.1f}".format(len(processed), angle), 
                            org=(cX,cY), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                            fontScale=1, 
                            color=(255,0,0), 
                            thickness=2)
                if (max_num_add is not None) and (len(processed) == max_num_add):
                        break
        except:
            pass
    assert len(processed) >= 1
    cv2.destroyAllWindows()

    # Save images for debugging. Then return the processed list.
    cv2.imshow("FINAL IMAGE before saving (PRESS ESC IF BAD).", image)
    key = cv2.waitKey(0)
    if key in utils.ESC_KEYS:
        sys.exit()
    cv2.imwrite(savedir, image)
    return processed


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
    trial = len([fname for fname in os.listdir(IMAGE_DIR) if 'left.png' in fname])
    trial_str = str(trial).zfill(3)
    tosave_txt = IMAGE_DIR+"img_"+trial_str+".txt"
    print("Inside image dir {}, this code will save as trial {}".format(IMAGE_DIR, trial))
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

    arm, _, d = utils.initializeRobots()
    utils.home(arm)
    print("current arm position: {}".format(arm.get_current_cartesian_position()))
    arm.close_gripper()

    # Let's do a pre-processing step where we first filter for contours and inspect.
    # This will also save the images for debugging purposes later.
    proc_left_contours  = get_good_contours(d.left_image_proc.copy(), 
                                            d.left_image.copy(), 
                                            d.get_left_bounds(),
                                            IMAGE_DIR+"img_"+trial_str+"_left.png",
                                            max_num_add=args.max_num_add)
    proc_right_contours = get_good_contours(d.right_image_proc.copy(), 
                                            d.right_image.copy(),
                                            d.get_right_bounds(),
                                            IMAGE_DIR+"img_"+trial_str+"_right.png",
                                            max_num_add=args.max_num_add)
    assert len(proc_left_contours) == len(proc_right_contours)
    assert len(proc_left_contours) <= args.max_num_add

    # NOW do the motion planning!
    motion_planning(d.left_image.copy(),
                    d.right_image.copy(),
                    proc_left_contours,
                    proc_right_contours,
                    tosave_txt, PARAMS, arm, d)
