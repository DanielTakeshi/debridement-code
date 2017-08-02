"""
I'll do the open-loop here for an experiment which DOES NOT use rotations. This --- version 03 --- uses
the flat tissue phantom that Steve gave me which is now at the correct height.

    (1) EIGHT seeds, must pick in some order (say right to left as usual).
    (2) There is NO BARRIER HERE. This is just to show some base case.
    (3) Ideally, we show that open loop fails, but simple behavioral cloning through time works. Don't
        use rotations here!
    (4) To be clear, WE ARE STILL DOING CALIBRATION. Once we move to a location via our matrix, we have
        to manually apply an adjustment, which happens in ONE step, where we move the (x,y) location to
        a better spot.
    (5) So repeat for all seeds. Thus, this demonstration involves EIGHT time steps.

Make sure I exit early if all eight seeds are not detected correctly!

I think perhaps success should also be measured based on how many seeds we actually can pick up. 
I'm not optimistic about all 8 getting picked up.

This is for collecting human demos AND open loop AND bcloning AND bcloning per time.
"""

import environ
from dvrk.robot import *
from autolab.data_collector import DataCollector
import cv2
import numpy as np
import os
import pickle
import sys

###########################
# ADJUST THESE AS NEEDED! #
###########################
ESC_KEY       = 27
TOPK_CONTOURS = 8 
COLLECT_DEMOS = False    # IMPORTANT!! True if we need human demos, false for test-time evaluation.
DEMO_TYPE     = 'bcloning_time'  # 'open' (i.e. no random forests), 'bcloning', or 'bcloning_time'.

IMDIR          = 'images/seeds_03'
RF_REGRESSOR   = 'config/daniel_final_mono_map_00.p'
DEMO_FILE_NAME = 'data/demos_seeds_03.p'
RANDOM_FORESTS = 'data/demos_seeds_03_maps.p'

# Requires some tweaking. NOTE: might be better to set vertical offset much lower than during
# real applications, because then we won't keep damaging our seeds!
ARM1_LCAM_HEIGHT = -0.16864652
EXTRA_HEIGHT     = 0.015
VERTICAL_OFFSET  = 0.012 # Use 0.008 for collecting demos, 0.012 for actual rollouts.
EXTRA_OFFSET     = {0: -0.001, 1: 0.000, 2: 0.002, 3: 0.002, 4: 0.004, 5: 0.004, 6: 0.005, 7: 0.005} # Don't ask. :-(
CLOSE_ANGLE      = 10


def get_num_stuff(filename):
    data = []
    f = open(filename,'r')
    num = 0
    while True:
        try:
            d = pickle.load(f)
            num += 1
        except EOFError:
            break
    return num


def save_images(d):
    """ For debugging/visualization. """
    cv2.imwrite(IMDIR+"left_proc.png",  d.left_image_proc)
    cv2.imwrite(IMDIR+"left_gray.png",  d.left_image_gray)
    #cv2.imwrite(IMDIR+"right_proc.png", d.right_image_proc)
    #cv2.imwrite(IMDIR+"right_gray.png", d.right_image_gray)


def call_wait_key(nothing=None):
    """ I have an ESC which helps me exit program. """
    key = cv2.waitKey(0)
    if key == ESC_KEY:
        print("Pressed ESC key. Terminating program...")
        sys.exit()


def show_images(d):
    """ For debugging/visualization. """
    #call_wait_key(cv2.imshow("Left Processed", d.left_image_proc))
    #call_wait_key(cv2.imshow("Left Gray",      d.left_image_gray))
    call_wait_key(cv2.imshow("Left BoundBox",  d.left_image_bbox))
    #call_wait_key(cv2.imshow("Left Circles",   d.left_image_circles))
    #print("Circles (left):\n{}".format(d.left_circles))


def initializeRobots(sleep_time=5):
    d = DataCollector()
    r1 = robot("PSM1") # left (but my right)
    r2 = robot("PSM2") # right (but my left)
    time.sleep(sleep_time)
    return (r1,r2,d)


def one_human_demonstration(places_to_visit, d, arm, ypred_arm_full):
    """ This will store a full trajectory for the eight sunflower seeds cases. We only need to 
    store the stuff where learning is done. OR ... just run the trajectory and we'll use the
    learned predictor. Depends on `COLLECT_DEMOS`.

    places_to_visit: [list] 
        List of tuples (cX,cY) indicating the pixels to which the arm should move.
    d: [DataCollector]
        The data collector.
    arm: [dvrk arm] 
        Either the left or right DVRK arm, from calling `robot("PSM{1,2}")`.
    ypred_arm_full: [np.array]
        Numpy array of shape (N,3) where N is the number of points to visit and each row
        consists of the x and y coordinates (in *robot* space) along with a fixed height.
        The height here is the target height PLUS a vertical offset, so that we move to a
        spot and then explicitly tell the robot arm to move downwards.
    """
    arm.home()
    assert ypred_arm_full.shape[0] == len(places_to_visit)
    arm.open_gripper(degree=90, time_sleep=2)
    default_rotation = (0.0, 0.0, -160.0)
    demo = []

    # Used if we're doing test-time rollouts and thus NOT collecting demonstrations.
    if not COLLECT_DEMOS:
        if DEMO_TYPE == 'bcloning' or DEMO_TYPE == 'bcloning_time':
            rand_forests = pickle.load(open(RANDOM_FORESTS))

    for i,pt_camera in enumerate(places_to_visit):
        arm_pt = ypred_arm_full[i]

        # (1) Get the arm to (the rough general area of) the target.
        print("\nMoving arm1 to PIXELS point {} indexed at {}".format(pt_camera, i))
        post, rott = (tuple(arm_pt), default_rotation)
        pos = [post[0], post[1], post[2]]

        # Note: rott contains the tuple, tfx.tb_angles means we have to call rot.yaw_deg, etc.
        rot = tfx.tb_angles(rott[0], rott[1], rott[2])
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.025)
        frame_before_moving = arm.get_current_cartesian_position()
        print("frame before moving: {}".format(frame_before_moving))

        if COLLECT_DEMOS:
            # (2) IMPORTANT. HUMAN-GUIDED STUFF HAPPENS HERE! 
            # If I need to abort, press ESC and this won't add to the pickle file.

            time.sleep(1)
            call_wait_key(cv2.imshow("DO XY MOVEMENT (always)", d.left_image_gray))
            cv2.destroyAllWindows()
            frame_after_xy_move = arm.get_current_cartesian_position()
            print("frame after xy moving: {}".format(frame_after_xy_move))
            demo.append((frame_before_moving, frame_after_xy_move, pt_camera, 'xy'))

        elif DEMO_TYPE == 'bcloning' or DEMO_TYPE == 'bcloning_time':
            # (2) Use our RANDOM_FORESTS to act as a guide to correct rotations and positioning.
            # MOVE to correct location, with height even, using same rotation! This might be the
            # one we started with or the one from the `if` case above.

            frame = arm.get_current_cartesian_position()
            robot_x, robot_y = frame.position[0], frame.position[1]
            if DEMO_TYPE == 'bcloning':
                result = rand_forests['all_seeds'].predict([[robot_x,robot_y]])
            elif DEMO_TYPE == 'bcloning_time':
                result = rand_forests['seed_'+str(i)].predict([[robot_x,robot_y]])
            result = np.squeeze(result) # So it's just (2,)

            pos = [result[0], result[1], post[2]]
            print("Current frame: {}".format(frame))
            print("predicted position: {}".format(pos))
            arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), SAFE_SPEED)

        # (3) Move gripper *downwards* to target object (ideally), using new frame.
        # Keep its (new) rotation, *not* the one I sent as the function input, if I changed it.
        # If doing the demo, we need to get new pos and new rotation. Otherwise, use old ones.

        frame = arm.get_current_cartesian_position()
        if COLLECT_DEMOS:
            pos = (frame.position[:3])
            rot = tfx.tb_angles(frame.rotation) 
        pos[2] -= (VERTICAL_OFFSET+EXTRA_OFFSET[i])
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), SAFE_SPEED)

        # (4) Close gripper and move *upwards*, ideally grabbing the object.
        arm.open_gripper(degree=CLOSE_ANGLE, time_sleep=2)
        pos[2] += (VERTICAL_OFFSET+EXTRA_OFFSET[i])
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), SAFE_SPEED)

        # (5) Finally, home the robot, then open the gripper to drop something (ideally).
        arm.home(open_gripper=False)
        arm.open_gripper(degree=90, time_sleep=2)

    # Store the demonstration --- a list of eight tuples --- if not using the regressor.
    arm.home()
    if COLLECT_DEMOS:
        f = open(DEMO_FILE_NAME, 'a')
        pickle.dump(demo, f)
        f.close()
        print("Dumped to file, now with {} demos stored.".format(get_num_stuff(DEMO_FILE_NAME)))


def motion_planning(contours_by_size, img, arm, arm_map):
    """ This gets everything _set_up_ for the motion planning problem.
    
    Parameters
    ----------
    contours_by_size: [list]
        A list of contours, arranged from largest area to smallest area.
    img: [np.array]
        Image the camera sees, in BGR form.
    arm: [dvrk arm]
        Represents the arm we're using for the dvrk.
    arm_map: [RandomForestRegressor]
        Maps from {left,right} camera's (cX,cY) to arm's position, assuming fixed height.
    """
    print("We identified {} contours but will keep top {}.".format(len(contours_by_size), TOPK_CONTOURS))
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

    # Collect only top K places to visit and insert ordering preferences. I do right to left,
    # but also I group into four rough columns and go bottom to top for all of them.
    places_to_visit = places_to_visit[:TOPK_CONTOURS]
    places_to_visit = sorted(places_to_visit, key=lambda x:x[0], reverse=True)
    num_points = len(places_to_visit)
    for i in range(0, len(places_to_visit), 2):
        point_i   = places_to_visit[i]
        point_ip1 = places_to_visit[i+1]
        if point_i[1] < point_ip1[1]:
            places_to_visit[i]   = point_ip1
            places_to_visit[i+1] = point_i

    # Number the places to visit in an image so I see them.
    for i,(cX,cY) in enumerate(places_to_visit):
        cv2.putText(img=img_for_drawing, text=str(i), org=(cX,cY), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(0,0,0), thickness=2)
        cv2.circle(img=img_for_drawing, center=(cX,cY), radius=3, color=(0,0,255), thickness=-1)
        print(i,cX,cY)

    # Show image with contours + exact centers. Exit if it's not looking good.
    index = len(os.listdir(IMDIR))
    cv2.imshow("Image with topK contours (exit if not looking good, index is {})".format(index), img_for_drawing)
    call_wait_key()
    if COLLECT_DEMOS:
        cv2.imwrite(IMDIR+"/im_"+str(index)+".png",  img_for_drawing)
    cv2.destroyAllWindows()

    # Manage predictions, store in `ypred_arm_full`.
    X = np.array(places_to_visit)
    ypred_arm = arm_map.predict(X)
    c = ARM1_LCAM_HEIGHT + EXTRA_HEIGHT
    ypred_arm_full = np.concatenate((ypred_arm, np.ones((num_points,1))*c), axis=1)
    print("\tHere's where we'll be visiting:\n{}".format(places_to_visit))
    print("\tin matrix form, X.shape is {} and elements are:\n{}".format(X.shape, X))
    print("\tfull ypred_arm1:\n{}".format(ypred_arm_full))

    # Perform ONE human-guided open-loop demonstration.
    one_human_demonstration(places_to_visit, d, arm, ypred_arm_full)


if __name__ == "__main__":
    """ See the top of the file for program-wide arguments. """
    assert DEMO_TYPE in ['open', 'bcloning', 'bcloning_time']
    arm1, _, d = initializeRobots(sleep_time=4)
    arm1.home()
    print("arm1 home: {}".format(arm1.get_current_cartesian_position()))
    #save_images(d)
    show_images(d)

    # Load the Random Forest regressor. It's just opening a silly pickle file.
    left_arm1_map = pickle.load(open(RF_REGRESSOR))
    motion_planning(d.left_contours_by_size, d.left_image, arm1, left_arm1_map)
