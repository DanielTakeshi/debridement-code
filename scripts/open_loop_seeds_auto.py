"""
Open loop for the case when we have a bunch of seeds scattered with random rotations.
This is designed to use data originally from the automatically collected trajectories.

Usage:

DNN+RF, pumpkins:
    python scripts/open_loop_seeds_auto.py --version_out 20 --max_num_add 8 --close_angle 30
DNN+RF, sunflowers:
    python scripts/open_loop_seeds_auto.py --version_out 21 --max_num_add 8 --close_angle 15
DNN, pumpkins:
    python scripts/open_loop_seeds_auto.py --version_out 22 --max_num_add 8 --close_angle 30 --no_rf_correctors
DNN, sunflowers:
    python scripts/open_loop_seeds_auto.py --version_out 23 --max_num_add 8 --close_angle 15 --no_rf_correctors
DNN+RF, pumpkins, v2 (now that I got the timing set up and better understanding of failure cases...):
    python scripts/open_loop_seeds_auto.py --version_out 24 --max_num_add 8 --close_angle 30
DNN+RF, raisins from Ken:
    python scripts/open_loop_seeds_auto.py --version_out 25 --max_num_add 8 --close_angle 30
DNN, raisins from Ken:
    python scripts/open_loop_seeds_auto.py --version_out 26 --max_num_add 8 --close_angle 30 --no_rf_correctors

Notes:

- Make sure I exit early if all eight seeds are not detected correctly!!
- Images should be saved of the initial contours to check for potential errors.
- Don't use `tee` as I should be writing to a file automatically.
- Use `--close_angle 30` for pumpkin, `--close_angle 15` for sunflower.
- Also, I've found that a z offset of -0.001 works well, maybe -0.002 for pumpkins?
- Use `--no_rf_correctors` to test a baseline of just the DNN. It _should_ be worse!
"""

import argparse
import cv2
import image_geometry
import numpy as np
import os
import pickle
import sys
import tfx
import time
import utilities as utils
from autolab.data_collector import DataCollector
from dvrk.robot import *
from keras.models import load_model
np.set_printoptions(suppress=True, linewidth=200)


def motion_planning(l_img, r_img, l_cnts, r_cnts, tosave_txt, PARAMS, arm, d, args):
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
    args: [Namespace]
        Arguments from the user, e.g. which contains offsets.
    """
    # Load the initial neural network. Don't forget the mean and standard deviation!!!
    f_network = load_model(PARAMS['nnet']['modeldir'])
    net_mean  = PARAMS['nnet']['X_mean']
    net_std   = PARAMS['nnet']['X_std']
    assert not os.path.exists(tosave_txt)
    text_file = open(tosave_txt, 'w')
    text_file.write("network mean: {}\n".format(net_mean))
    text_file.write("network std:  {}\n".format(net_std))

    # Try to make this over where a bin/container is located to hold the dropped seeds.
    bin_position = [-0.025, 0.06, -0.125]
    start_yaw = 0
    start_pitch, start_roll = utils.get_interpolated_pitch_and_roll(start_yaw)
    utils.move(arm, bin_position, [start_yaw,start_pitch,start_roll], args.speed_class)
    arm.open_gripper(degree=100, time_sleep=2)

    # Important! We want to measure time. But don't include start-up time above.
    overall_start_time = time.time()

    for (left,right) in zip(l_cnts, r_cnts):
        # Given left/right points, figure out the desired robot points.
        lx,ly,langle,lyaw = left
        rx,ry,rangle,ryaw = right
        camera_pt = np.array( utils.camera_pixels_to_camera_coords([lx,ly], [rx,ry]) )

        # Average out the yaw values. Use those to get rotation and then input to network.
        yaw = (lyaw+ryaw)/2.
        pitch, roll = utils.get_interpolated_pitch_and_roll(yaw)
        rotation = np.array([yaw, pitch, roll])
        net_input = (np.concatenate((camera_pt, rotation))).reshape((1,6))
        text_file.write("\nnet_input (camera_pt, rot): {}\n".format(net_input)) 

        # Use the network here!! As usual the leading dimension is the "batch" size.
        net_input = (net_input - net_mean) / net_std
        predicted_pos = np.squeeze(f_network.predict(net_input)).tolist()
        text_file.write("pred_robot_pos (no offsets): {}\n".format(['%4f'%elem for elem in predicted_pos]))
        assert net_input.shape == (1,6)
        assert len(predicted_pos) == 3

        # Apply the random forest corrector depending on the yaw. Yeah I know it's ugly code...
        if args.no_rf_correctors:
            text_file.write("\t(No RF corrector)\n")
        else:
            if rotation[0] < -67.5: 
                yaw_key = -90
            elif rotation[0] < -22.5: 
                yaw_key = -45
            elif rotation[0] <  22.5: 
                yaw_key =   0
            elif rotation[0] <  67.5: 
                yaw_key =  45
            else:
                yaw_key =  90
            residual_vec = np.squeeze( PARAMS[yaw_key].predict([predicted_pos]) )    
            predicted_pos = predicted_pos - residual_vec
            text_file.write("\tresidual vector from RF corrector: {}\n".format(residual_vec))
            text_file.write("\trevised predicted_pos (before offsets): {}\n".format(predicted_pos))

        # -------------------------------------------
        # NOW we can move. Gaaah. Apply offsets here.
        # -------------------------------------------
        predicted_pos[0] += args.x_offset
        predicted_pos[1] += args.y_offset
        predicted_pos[2] += args.z_offset
        predicted_pos[2] += args.zoffset_safety

        # First rotate at the bin position, then move to the seed (with rotation fixed).
        utils.move(arm, bin_position, rotation, args.speed_class)
        utils.move(arm, predicted_pos, rotation, args.speed_class)

        predicted_pos[2] -= args.zoffset_safety
        utils.move(arm, predicted_pos, rotation, args.speed_class)
        arm.open_gripper(degree=args.close_angle, time_sleep=1.5) # Experimentally, 1.5 seems to work.
        #arm.open_gripper(degree=args.close_angle) # Doesn't work, it goes up as it closes. :-(
        predicted_pos[2] += args.zoffset_safety
        utils.move(arm, predicted_pos, rotation, args.speed_class)
        time.sleep(0.5)

        utils.move(arm, bin_position, rotation, args.speed_class)
        #arm.open_gripper(degree=100, time_sleep=1)
        arm.open_gripper(degree=100) # This works! :-)

    overall_elapsed_time = time.time() - overall_start_time
    text_file.write("overall_elapsed_time: {}\n".format(overall_elapsed_time))
    print("overall_elapsed_time: {}".format(overall_elapsed_time))
    text_file.close()


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
                cv2.circle(img=image, center=(cX,cY), radius=5, color=(0,0,255), thickness=-1)
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
    pp.add_argument('--close_angle', type=int, default=30) 
    pp.add_argument('--guidelines_dir', type=str, default='traj_collector/guidelines.p')
    pp.add_argument('--max_num_add', type=int, default=35) # If I do 36, I'll be restarting a lot. :-)
    pp.add_argument('--no_rf_correctors', action='store_true')
    pp.add_argument('--speed_class', type=str, default='Fast')
    pp.add_argument('--x_offset', type=float, default=0.000)
    pp.add_argument('--y_offset', type=float, default=0.000)
    pp.add_argument('--z_offset', type=float, default=-0.002) # Tweak this value!!
    pp.add_argument('--zoffset_safety', type=float, default=0.006) # And this one ...
    args = pp.parse_args()

    # Check the image versions, etc.
    IN_VERSION  = str(args.version_in).zfill(2)
    OUT_VERSION = str(args.version_out).zfill(2)
    IMAGE_DIR   = 'images/seeds_v'+OUT_VERSION+'/'
    assert (OUT_VERSION is not None) and (IN_VERSION is not None)
    assert args.speed_class in ['Slow', 'Fast']

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

    # Initialize, etc...
    arm, _, d = utils.initializeRobots()
    arm.close_gripper()

    # Let's do a pre-processing step where we first filter for contours and inspect.
    # This will also save the images for debugging purposes later.
    proc_right_contours = get_good_contours(d.right_image_proc.copy(), 
                                            d.right_image.copy(),
                                            d.get_right_bounds(),
                                            IMAGE_DIR+"img_"+trial_str+"_right.png",
                                            max_num_add=args.max_num_add)
    proc_left_contours  = get_good_contours(d.left_image_proc.copy(), 
                                            d.left_image.copy(), 
                                            d.get_left_bounds(),
                                            IMAGE_DIR+"img_"+trial_str+"_left.png",
                                            max_num_add=args.max_num_add)
    assert len(proc_left_contours) == len(proc_right_contours)
    assert len(proc_left_contours) <= args.max_num_add

    # NOW do the motion planning!
    motion_planning(d.left_image.copy(),
                    d.right_image.copy(),
                    proc_left_contours,
                    proc_right_contours,
                    tosave_txt, PARAMS, arm, d, args)
