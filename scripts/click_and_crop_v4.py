""" 
This will be for measuring the errors with the NON-CALIBRATED baseline. I actually fit
a z-plane to this to reduce z-errors, but the x and y errors should still be substantial.
BE CAREFUL TO ACTUALLY DRAG SOMETHING, if you click once it will cause an assertion error.

Example call (make sure the numbers such as 50 all match the corresponding fixed_yaw value):

clear; clear; rm -r images/check_regressors_v50/; rm config/calibration_results/data_v50.p; \
python scripts/click_and_crop_v4.py --version_out 50 --max_num_add 36 --fixed_yaw -90

(c) September 2017 by Daniel Seita
"""

import argparse
import cv2
import image_geometry
import numpy as np
import os
import pickle
import sys
import utilities as utils
from autolab.data_collector import DataCollector
from dvrk.robot import *
from keras.models import load_model
np.set_printoptions(suppress=True)

# For `click_and_crop`.
POINTS          = []
CENTER_OF_BOXES = []


def get_z_from_xy_values(info, x, y):
    """ We fit a plane. """
    return (info['z_alpha'] * x) + (info['z_beta'] * y) + info['z_gamma']


def get_key_from_rot(rotation):
    """ Yeah I know it's ugly code... """
    if rotation < -67.5: 
        return -90
    elif rotation < -22.5: 
        return -45
    elif rotation <  22.5: 
        return 0
    elif rotation <  67.5: 
        return 45
    else:
        return 90


def click_and_crop(event, x, y, flags, param):
    global POINTS, CENTER_OF_BOXES
             
    # If left mouse button clicked, record the starting (x,y) coordinates 
    # and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS.append((x,y))
                                                 
    # Check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # Record ending (x,y) coordinates and indicate that cropping is finished AND save center!
        POINTS.append((x,y))

        upper_left = POINTS[-2]
        lower_right = POINTS[-1]
        assert upper_left[0] < lower_right[0]
        assert upper_left[1] < lower_right[1]
        center_x = int(upper_left[0] + (lower_right[0]-upper_left[0])/2)
        center_y = int(upper_left[1] + (lower_right[1]-upper_left[1])/2)
        CENTER_OF_BOXES.append( (center_x,center_y) )
        
        # Draw a rectangle around the region of interest, w/center point. Blue=Before, Red=AfteR.
        cv2.rectangle(img=updated_image_copy, 
                      pt1=POINTS[-2], 
                      pt2=POINTS[-1], 
                      color=(0,0,255), 
                      thickness=2)
        cv2.circle(img=updated_image_copy, 
                   center=CENTER_OF_BOXES[-1], 
                   radius=6, 
                   color=(0,0,255), 
                   thickness=-1)
        cv2.putText(img=updated_image_copy, 
                    text="{}".format(CENTER_OF_BOXES[-1]), 
                    org=CENTER_OF_BOXES[-1],  
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, 
                    color=(0,0,255), 
                    thickness=2)
        cv2.imshow("This is in the click and crop method AFTER the rectangle. "+
                   "(Press any key, or ESC If I made a mistake.)", updated_image_copy)


def get_good_contours(image, image_contours, max_num_add):
    contours = []
    num_added = 0

    for i, (cX,cY,approx,peri) in enumerate(image_contours):  
        if utils.filter_point(cX,cY, xlower=500, xupper=1500, ylower=50, yupper=1000):
            continue
        cv2.circle(img=image, center=(cX,cY), radius=50, color=(0,0,255), thickness=1)
        cv2.circle(img=image, center=(cX,cY), radius=4, color=(0,0,255), thickness=-1)
        cv2.drawContours(image=image, contours=[approx], contourIdx=-1, color=(0,255,0), thickness=3)

        name = "ESC if duplicate/undesired, other key to proceed."
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 2000, 4000)
        cv2.imshow(name, image)
        firstkey = cv2.waitKey(0) 

        # Update `contours` and add extra text to make the image updating easier to follow.
        if firstkey not in utils.ESC_KEYS:
            contours.append( (cX,cY,approx,peri) )
            num_added += 1
            cv2.putText(img=image,
                        text="{}".format(num_added), 
                        org=(cX,cY), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, 
                        color=(255,0,0), 
                        thickness=2)
        if num_added == max_num_add:
            break
    return contours


def sample_rotation(fixed_yaw):
    if fixed_yaw is not None:
        yaw = fixed_yaw
    else:
        yaw = np.random.uniform(low=-90, high=90)
    pitch, roll = utils.get_interpolated_pitch_and_roll(yaw)
    return np.array([yaw, pitch, roll])


if __name__ == "__main__":
    """ Set up a configuration here. See `images/README.md`. Use 2X for neural net stuff. """
    pp = argparse.ArgumentParser()
    pp.add_argument('--version_in', type=int, default=1, help='For now, it\'s 0')
    pp.add_argument('--version_out', type=int, help='See `images/README.md`.')
    pp.add_argument('--x_offset', type=float, default=0.000)
    pp.add_argument('--y_offset', type=float, default=0.000)
    pp.add_argument('--z_offset', type=float, default=0.000)
    pp.add_argument('--fixed_yaw', type=float, help='If not provided, yaw randomly chosen in [-90,90]')
    pp.add_argument('--max_num_add', type=int, default=35) # If I do 36, I'll be restarting a lot. :-)
    pp.add_argument('--guidelines_dir', type=str, default='traj_collector/guidelines.p')
    pp.add_argument('--use_rf_correctors', action='store_true')
    pp.add_argument('--use_rigid_body', action='store_true')
    args = pp.parse_args()

    # Just because I think v1 is now better ... but IDK. Can comment out if desired.
    assert args.version_in != 0, "Are you sure you want version 0?"

    IN_VERSION  = str(args.version_in).zfill(2)
    OUT_VERSION = str(args.version_out).zfill(2)
    OUTPUT_FILE = 'config/calibration_results/data_v'+OUT_VERSION+'.p'
    IMDIR       = 'images/check_regressors_v'+OUT_VERSION+'/'
    assert (OUT_VERSION is not None) and (IN_VERSION is not None)
    # To prevent overriding
    assert not os.path.exists(OUTPUT_FILE), "OUTPUT_FILE: {}".format(OUTPUT_FILE)
    assert not os.path.exists(IMDIR), "IMDIR: {}".format(IMDIR)
    os.makedirs(IMDIR)
    PARAMS = pickle.load(
            open('config/mapping_results/auto_params_matrices_v'+IN_VERSION+'.p', 'r')
    )
    rotation = sample_rotation(args.fixed_yaw)
    print("Our starting rotation: {}".format(rotation))
    
    # Now get the robot arm initialized and test it out!
    arm, _, d = utils.initializeRobots()
    print("current arm position: {}".format(arm.get_current_cartesian_position()))
    utils.home(arm, rot=rotation)
    print("current arm position: {}".format(arm.get_current_cartesian_position()))
    arm.close_gripper()
    cv2.imwrite("images/left_image.jpg", d.left_image)
    utils.call_wait_key(cv2.imshow("left camera, used for contours", d.left_image_proc))

    ## ---------------------------------------------------------------------------------------
    ## PART ONE: Get the contours. Manual work here to cut down on boredom in the second part.
    ## Actually ... we don't even need both images, right? I mean, c'mon...
    ## ---------------------------------------------------------------------------------------
    left_im = d.left_image.copy()
    left_c = get_good_contours(left_im, d.left_contours, args.max_num_add)
    cv2.destroyAllWindows()

    ## ---------------------------------------------------------------------------------------
    ## PART TWO: Explicitly fit the point to the upper left corner of the grid. This is
    ## definitely cheating, like it is to cheat to have that z-coordinate, but hey, baselines!
    ## Oh, load the z-plane information here in the `info` dictionary.
    ## ---------------------------------------------------------------------------------------
    info = pickle.load( open(args.guidelines_dir,'r') )
    utils.call_wait_key(cv2.imshow("Left camera (move to upper right corner now!)", d.left_image))
    pos, rot = utils.lists_of_pos_rot_from_frame( arm.get_current_cartesian_position() )
    rotation = sample_rotation(args.fixed_yaw)
    utils.move(arm, pos, rotation, 'Fast')
    utils.call_wait_key(cv2.imshow("Now re-adjust with correct rotation", d.left_image))
    start_pos, start_rot = utils.lists_of_pos_rot_from_frame( arm.get_current_cartesian_position() )
    print("\nTouched the upper left corner.\nStart_Pos: {}\nStart_Rot: {}".format(start_pos, start_rot))
    print("Note that the sampled rotation was: {}".format(rotation))

    ## ---------------------------------------------------------------------------------------
    ## PART THREE: iterate through contours and drag the box around the end-effector location.
    ## I had to measure the grid but it seems like 16mm wide in one row, and vertically 8mm.
    ## E.g. 01 to 02 to 11 to 10 forms a square which has edges (from circle _centers_) 16mm.
    ## Sorry, the below grid is 1-indexed, bleh.
    ## 
    ##     01  02  03  04  05
    ##       06  07  08  09
    ##     10  11  12  13  14
    ##       15  16  17  18
    ##     19  20  21  22  23
    ##       24  25  26  27
    ##     28  29  30  31  32
    ##       33  34  35  36
    ## ---------------------------------------------------------------------------------------
    xoffset = 0
    yoffset = 0
    for ii, l_cnt in enumerate(left_c):
        lx,ly,_,_ = l_cnt

        # Figure out interpolated x and y values. Get z-coord from fitted plane.
        xcoord = start_pos[0] + xoffset
        ycoord = start_pos[1] + yoffset
        zcoord = get_z_from_xy_values(info, xcoord, ycoord)
        predicted_pos = [xcoord, ycoord, zcoord]
        assert len(predicted_pos) == 3
 
        # After fixing `predicted_pos` we'll get xoffset and yoffsets updated. Yeah, a bit convoluted.
        index = ii+1 # Because the diagram above in the comments is one-indexed, lol...
        if index in [5,9,14,18,23,27,32]:
            yoffset += 8
            if index in [5,14,23,32]:
                xoffset = 8
            else:
                xoffset = 0
        else:
            xoffset += 16

        # Robot moves to that point and will likely be off. Let the camera refresh.
        utils.move(arm, predicted_pos, rotation, 'Slow')
        time.sleep(5)

        # Update image (left, though it doesn't matter) and put center coords. Blue=Before, Red=AfteR.
        updated_image_copy = (d.left_image).copy()
        cv2.circle(img=updated_image_copy, center=(lx,ly), radius=6, color=(255,0,0), thickness=-1)
        cv2.putText(img=updated_image_copy, 
                    text="{}".format((lx,ly)), 
                    org=(lx,ly), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, 
                    color=(255,0,0), 
                    thickness=2)

        # Now we apply the callback and drag a box around the end-effector on the (updated!) image.
        position = arm.get_current_cartesian_position()
        window_name = "Moved to pred_pos incl. offset {}. Click + drag box on end-eff.".format(predicted_pos)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) # Oh, the _name_ is recorded...!
        cv2.resizeWindow(window_name, 2000, 4000)
        cv2.setMouseCallback(window_name, click_and_crop) # Clicks to this window are recorded!
        cv2.imshow(window_name, updated_image_copy)
        key = cv2.waitKey(0)

        # Now save the image with the next available index. It will contain the contours.
        index = len(os.listdir(IMDIR))
        cv2.imwrite(IMDIR+"/point_"+str(index).zfill(2)+".png", updated_image_copy)
        cv2.destroyAllWindows()

        # Get position and orientation of the arm, save (in a dictionary), & reset. 
        # I.e. the target_pos is what the random forest predicted for the target position.
        new_pos, new_rot = utils.lists_of_pos_rot_from_frame(
                arm.get_current_cartesian_position()
        )
        data_pt = {'target_pos': predicted_pos, # Includes offset, not sure if I should reset those...
                   'target_rot': rotation,
                   'actual_pos': new_pos,
                   'actual_rot': new_rot,
                   'center_target_pixels': (lx,ly),
                   'center_actual_pixels': CENTER_OF_BOXES[-1]}
        utils.storeData(OUTPUT_FILE, data_pt)

        # Some stats for debugging, etc. This time, `i` really is `num_added`.
        print("contour {}, data_pt: {}".format(ii, data_pt))
        assert (2*(ii+1)) == (2*len(CENTER_OF_BOXES)) == len(POINTS)
        utils.home(arm, rot=rotation)

        # Update the rotation (if using fixed yaw, this uses the same exact rotation).
        rotation = sample_rotation(args.fixed_yaw)
        utils.home(arm, rot=rotation)
        time.sleep(1)

    utils.home(arm, rot=rotation)
    cv2.destroyAllWindows()
