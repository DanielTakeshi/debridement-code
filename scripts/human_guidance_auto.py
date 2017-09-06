""" 
Use this after training a neural network to map from camera points and *desired* yaw, 
pitch, and roll to a set of robot points. There is some systematic errors that I've 
shown from `click_and_crop_v3.py`. Hence, this will do the human guided stuff so 
that in the end we have a dataset which we can then train something to go from:

    (rx,ry,rz,yaw,pitch,roll) -> (change_x, change_y, change_z)

Though:

    - For the actual function, we should ignore pitch and roll since its known here
      from interpolation. This is *unlike* what I did in the automatic trajectory
      collection code.
    - I am not going to model a "change" in yaw, pitch, and roll since that will be
      too hard to manipulate with human fingers.

Run this with argparse command lines. See `scripts/click_and_crop_v3.py` for details.
It uses a similar form as that because we have to check for the correct contours, etc.

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


def get_good_contours(image, image_contours, max_num_add):
    """ Exactly the same as in `scripts.click_and_crop_v3.py`. """
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
    """ Exactly the same as in `scripts.click_and_crop_v3.py`. """
    if fixed_yaw is not None:
        yaw = fixed_yaw
    else:
        yaw = np.random.uniform(low=-90, high=90)
    pitch, roll = utils.get_interpolated_pitch_and_roll(yaw)
    return np.array([yaw, pitch, roll])


if __name__ == "__main__":
    """ 
    Set up a configuration here. See `images/README.md`. Use 2X output for neural net stuff. 
    The idea is that we might be able to combine these output files together later.
    Also, these arguments are very similar to those in `scripts/click_and_crop_v3.py`. 
    """
    pp = argparse.ArgumentParser()
    pp.add_argument('--version_in', type=int, default=0, help='For now, it\'s 0')
    pp.add_argument('--version_out', type=int, help='See `images/README.md`.')
    pp.add_argument('--x_offset', type=float, default=0.000)
    pp.add_argument('--y_offset', type=float, default=0.000)
    pp.add_argument('--z_offset', type=float, default=0.000) # Keep at zero for now.
    pp.add_argument('--fixed_yaw', type=float, help='If not provided, yaw randomly chosen in [-90,90]')
    pp.add_argument('--max_num_add', type=int, default=35) # If I do 36, I'll be restarting a lot. :-)
    pp.add_argument('--guidelines_dir', type=str, default='traj_collector/guidelines.p')
    args = pp.parse_args()

    # Just to check if I want 0 or 1 ...
    assert args.version_in != 0, "Just to be clear, you want version in to be 0?"

    IN_VERSION  = str(args.version_in).zfill(2)
    OUT_VERSION = str(args.version_out).zfill(2)
    OUTPUT_FILE = 'config/calibration_results/data_human_guided_v'+OUT_VERSION+'.p'
    assert (OUT_VERSION is not None) and (IN_VERSION is not None)
    assert not os.path.exists(OUTPUT_FILE) # To prevent overriding
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
    
    # The right image will typically tell us how many contours we can expect (34, 35, 36??).
    cv2.imwrite("images/right_image.jpg", d.right_image)
    cv2.imwrite("images/left_image.jpg", d.left_image)
    utils.call_wait_key(cv2.imshow("RIGHT camera, used for contours", d.right_image_proc))
    utils.call_wait_key(cv2.imshow("left camera, used for contours", d.left_image_proc))

    # Don't forget to load our trained neural network!
    print("Neural Network model directory: {}".format(PARAMS['modeldir']))
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
 
    # --------------------------------------------------------------------------------------
    # PART TWO: iterate through contours and tell the robot to move based on the neural net.
    # It will likely be off. Thus do human correction! To get changes in x, y, and z values.
    # Some subtlety here. We want the input to be the _predicted_ robot point (+yaw, etc.),
    # NOT the actual robot point. We will know the predicted one, but the actual point is 
    # very noisy. We do, however, need the actual point to get correct workspace offsets!!!!
    # --------------------------------------------------------------------------------------
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

        # Use offsets. (Edit: I don't think we want offsets ... set them to zero.)
        predicted_pos[0] += args.x_offset
        predicted_pos[1] += args.y_offset
        predicted_pos[2] += args.z_offset

        # Robot moves to that point and is likely off. Let the camera refresh. Don't move until then!
        utils.move(arm, predicted_pos, rotation, 'Slow')
        time.sleep(4)

        # This should NOT be the input to the RF! We should use `pos` or `target` instead.
        # BUT we still need it to get correct offsets in the _workspace_ (i.e. real world)!!
        actual_pos_after_moving = arm.get_current_cartesian_position() 

        # IMPORTANT! Now I do some human motion!!
        updated_image_copy = (d.left_image).copy()
        window_name = "Tried moving to {}, actual pos {} ADJUST IT, then press space bar".format(
                predicted_pos, actual_pos_after_moving)
        cv2.imshow(window_name, updated_image_copy)
        key = cv2.waitKey(0)
        if key in utils.ESC_KEYS:
            print("Pressed ESC. Terminating program ...")
            sys.exit()
        cv2.destroyAllWindows()

        # Now record the new position of the arm (after human correction), and save it.
        # These will, of course, save all the relevant (yaw, pitch, roll) values.
        new_pos = arm.get_current_cartesian_position()
        data_pt = {'original_robot_point_prediction':          predicted_pos,
                   'measured_robot_point_before_human_change': actual_pos_after_moving, 
                   'measured_robot_point_after_human_change':  new_pos}
        utils.storeData(OUTPUT_FILE, data_pt)

        # Some stats for debugging, etc. This time, `i` really is `num_added`.
        print("contour index {}, data_pt: {}".format(ii, data_pt))
        utils.home(arm, rot=rotation)

        # Update the rotation (if using fixed yaw, this uses the same exact rotation).
        rotation = sample_rotation(args.fixed_yaw)
        utils.home(arm, rot=rotation)
        time.sleep(1)

    utils.home(arm, rot=rotation)
    cv2.destroyAllWindows()
