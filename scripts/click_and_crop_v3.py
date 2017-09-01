""" 
Because I need to detect mouse clicks so that I can find and visualize where the rigid body
and random forests are going wrong. AFTER calibration, I will send the robot end effector to 
many points. Then, once it goes to each point, this code will stop, show an _image_ of the 
target point and the actual robot points. Then I'll drag a box around where the end-effector 
is, and then the code should automatically record stuff. Be careful to record the right stuff,
so it's manual but I'm usually good with this.

BE CAREFUL TO ACTUALLY DRAG SOMETHING, if you click once it will cause an assertion error.

THIS (version 3) IS FOR THE AUTOMATIC TRAJECTORY CASE. Use version 2 for the manual stuff.

Example call:

clear; clear; rm -r images/check_regressors_v23/; rm config/calibration_results/data_v23.p; \
python scripts/click_and_crop_v3.py --version_out 23 \
                                    --max_num_add 35 \
                                    --z_offset 0.0 \
                                    --fixed_yaw -45

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
        cv2.imshow("ESC if duplicate/undesired, other key to proceed. Added {}".format(num_added), image)
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
    """ Set up a configuration here. SEE `images/README.md`. Use 2X for neural net stuff. """

    pp = argparse.ArgumentParser()
    pp.add_argument('--version_in', type=int, default=0, help='For now, it\'s 0')
    pp.add_argument('--version_out', type=int, help='SEE `images/README.md`.')
    pp.add_argument('--x_offset', type=float, default=0.000)
    pp.add_argument('--y_offset', type=float, default=0.000)
    pp.add_argument('--z_offset', type=float, default=0.002)
    pp.add_argument('--fixed_yaw', type=float, help='If not provided, yaw randomly chosen in [-90,90]')
    pp.add_argument('--max_num_add', type=int, default=35) # If I do 36, I'll be restarting a lot. :-)
    pp.add_argument('--simulate_right', type=int, default=0, help='1 if simulate right, 0 if false')
    pp.add_argument('--guidelines_dir', type=str, default='traj_collector/guidelines.p')
    args = pp.parse_args()

    IN_VERSION  = str(args.version_in).zfill(2)
    OUT_VERSION = str(args.version_out).zfill(2)
    OUTPUT_FILE = 'config/calibration_results/data_v'+OUT_VERSION+'.p'
    IMDIR       = 'images/check_regressors_v'+OUT_VERSION+'/'
    assert (OUT_VERSION is not None) and (IN_VERSION is not None)
    assert not os.path.exists(OUTPUT_FILE) # To prevent overriding
    assert not os.path.exists(IMDIR) # To prevent overriding
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

    ## ------------------------------------------------------------------------------------
    ## This will test calibration! Two main cases: either we simulate the right camera's
    ## points from the left camera, or we track both of the cameras. I think tracking with
    ## both cameras is best, so the easiest way is to go through all the right contours but
    ## just track the ones that work, so progressively accumulate 36 (or 35...) values. Do
    ## the same for the left image. Then, with this data, we redo the entire process but we
    ## know we no longer have to do the repeated skipping I was doing.
    ##
    ## Note: this used to be its own method, but I got confused about the callback code, 
    ## and I'm in a bit of a rush, so sorry!
    ## ------------------------------------------------------------------------------------

    # The right image will tell us how many contours we can expect (34, 35, 36??).
    cv2.imwrite("images/right_image.jpg", d.right_image)
    cv2.imwrite("images/left_image.jpg", d.left_image)
    utils.call_wait_key(cv2.imshow("RIGHT camera, used for contours", d.right_image_proc))
    utils.call_wait_key(cv2.imshow("left camera, used for contours", d.left_image_proc))

    # Don't forget to load our trained neural network!
    f_network = load_model(PARAMS['modeldir'])
    net_mean  = PARAMS['X_mean']
    net_std   = PARAMS['X_std']

    # PART ONE: Get the contours. Manual work here to cut down on boredom in the second part.
    right_im = d.right_image.copy()
    left_im = d.left_image.copy()
    right_c = get_good_contours(right_im, d.right_contours, args.max_num_add)
    cv2.destroyAllWindows()
    left_c = get_good_contours(left_im, d.left_contours, args.max_num_add)
    cv2.destroyAllWindows()
    assert len(right_c) == len(left_c)
 
    # PART TWO: iterate through contours and drag the box around the end-effector location.
    for ii, (l_cnt, r_cnt) in enumerate(zip(left_c, right_c)):
        lx,ly,_,_ = l_cnt
        rx,ry,_,_ = r_cnt
        camera_pt = np.array( utils.camera_pixels_to_camera_coords([lx,ly], [rx,ry]) )
        net_input = (np.concatenate((camera_pt, rotation))).reshape((1,6))

        # Use the network here!! As usual the leading dimension is the "batch" size.
        net_input = (net_input - net_mean) / net_std
        predicted_pos = np.squeeze(f_network.predict(net_input)).tolist()
        print("\ncamera_pt: {}  pred_robot_pos (no offsets): {}".format(
            ['%.4f'%elem for elem in camera_pt], ['%4f'%elem for elem in predicted_pos]))
        assert net_input.shape == (1,6)
        assert len(predicted_pos) == 3

        # Use offsets.
        predicted_pos[0] += args.x_offset
        predicted_pos[1] += args.y_offset
        predicted_pos[2] += args.z_offset

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
        cv2.namedWindow(window_name)
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
    arm.close_gripper()
    cv2.destroyAllWindows()
