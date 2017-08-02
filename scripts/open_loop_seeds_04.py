"""
I'll do the open-loop here for an experiment which involves ROTATIONS. This --- version 04 --- uses
the flat tissue phantom that Steve gave me which ALSO has the RAISED BARRIER.

    (1) EIGHT seeds, must pick in some order (say right to left as usual).
    (2) There IS A BARRIER here. We will have to enforce two rotations!! They are at the 5th and 6th
        out of the eight seeds, so if zero-indexing, thery're the 4th and 5th indices. I do these from
        right to left so it doesn't matter which of the 4th or 5th indices they use.
    (3) Ideally, we show that open loop fails badly, but simple behavioral cloning through time works
        to a somewhat acceptable level. 
    (4) To be clear, WE ARE STILL DOING CALIBRATION. Once we move to a location via our matrix, we have
        to manually apply an adjustment. For six of these there is one (x,y) movement. For two others
        (and we know which of those two, since I move from right to left) I ROTATE FIRST, then move. It
        is much easier to do it that way rather than the other way around.
    (5) So repeat for all seeds. Thus, each human demonstration involves TEN time steps.

REMINDER: do rotations then move. So for stuff other than indices 4 and 5, I should hit the space bar 
FIRST, then move! Obviously, make sure I exit early if all eight seeds are not detected correctly!
"""

import environ
from dvrk.robot import *
from autolab.data_collector import DataCollector
import cv2
import numpy as np
import pickle
import sys

###########################
# ADJUST THESE AS NEEDED! #
###########################
ESC_KEY       = 27
TOPK_CONTOURS = 8 
COLLECT_DEMOS = True    # IMPORTANT!! True if we need human demos, false for test-time evaluation.
DEMO_TYPE     = 'open'  # 'open' (i.e. no random forests), 'bcloning', or 'bcloning_time'.

IMDIR          = 'images/seeds_04'
RF_REGRESSOR   = 'config/daniel_final_mono_map_00.p'
DEMO_FILE_NAME = 'data/demos_seeds_04.p'
RANDOM_FORESTS = 'data/demos_seeds_04_maps.p'

# Requires some tweaking. NOTE: might be better to set vertical offset much lower than during
# real applications, because then we won't keep damaging our seeds!
ARM1_LCAM_HEIGHT = -0.16864652
EXTRA_HEIGHT     = 0.015
VERTICAL_OFFSET  = 0.008 # Use 0.008 for collecting demos.
ROTATION_INDICES = [4, 5]


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
        rf_rot, rf_xy, rf_rot_camera, rf_xy_camera = pickle.load(open(RANDOM_FORESTS))

    for i,pt_camera in enumerate(places_to_visit):
        arm_pt = ypred_arm_full[i]

        # (1) Get the arm to (the rough general area of) the target.
        print("\nMoving arm1 to PIXELS point {} indexed at {}".format(pt_camera, i))
        post, rott = (tuple(arm_pt), default_rotation)
        pos = [post[0], post[1], post[2]]

        # Note: rott contains the tuple, tfx.tb_angles means we have to call rot.yaw_deg, etc.
        rot = tfx.tb_angles(rott[0], rott[1], rott[2])

        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.02)
        frame_before_rotation = arm.get_current_cartesian_position()
        print("frame before rotation: {}".format(frame_before_rotation))

        if COLLECT_DEMOS:
            # (2) IMPORTANT. HUMAN-GUIDED STUFF HAPPENS HERE! 
            # If I need to abort, press ESC and this won't add to the pickle file.

            # (2a) ROTATE the end-effectors but don't move anything else. Then press any key.
            # If we don't need to rotate, press any key but just don't add to the demo list.
            time.sleep(1)
            call_wait_key(cv2.imshow("DO ROTATION (if necessary, else just press key other than escape)", d.left_image_gray))
            cv2.destroyAllWindows()
            frame_before_moving = arm.get_current_cartesian_position()
            print("frame before moving: {}".format(frame_before_moving))
            if i in ROTATION_INDICES:
                demo.append((frame_before_rotation, frame_before_moving, pt_camera, 'rotation'))

            # (2b) MOVE to correct location, keeping height as even as possible. Then press any key.
            time.sleep(1)
            call_wait_key(cv2.imshow("DO XY MOVEMENT (always)", d.left_image_gray))
            cv2.destroyAllWindows()
            frame_after_xy_move = arm.get_current_cartesian_position()
            print("frame after xy moving: {}".format(frame_after_xy_move))
            demo.append((frame_before_moving, frame_after_xy_move, pt_camera, 'xy'))

        else:
            # (2) Use our RANDOM_FORESTS to act as a guide to correct rotations and positioning.

            # (2a) ROTATE the end-effectors, without moving (x,y,z), IF on third or fourth seed.
            # Here, `rot` already contains what we want since we haven't rotated from home position.
            # Be sure to use the same position, specified in `pos`. Keep xy and rotations separate!
            if i in ROTATION_INDICES:
                pred_rot = rf_rot.predict([[rot.yaw_deg, rot.pitch_deg, rot.roll_deg]])
                pred_rot = np.squeeze(pred_rot) # So it's just (3,)
                rot = tfx.tb_angles(pred_rot[0], pred_rot[1], pred_rot[2])
                print("Current frame: {}".format(frame_before_rotation))
                print("pred_rot: {} and rot: {}".format(pred_rot, rot))
                arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), SAFE_SPEED)

            # (2b) MOVE to correct location, with height even, using same rotation!
            # This might be the one we started with or the one from the `if` case above.
            frame = arm.get_current_cartesian_position()
            robot_x, robot_y = frame.position[0], frame.position[1]
            result = rf_xy.predict([[robot_x,robot_y]])
            result = np.squeeze(result) # So it's just (2,)
            new_x, new_y = result[0], result[1]
            pos = [new_x, new_y, post[2]]
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
        pos[2] -= VERTICAL_OFFSET
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), SAFE_SPEED)

        # (4) Close gripper and move *upwards*, ideally grabbing the object.
        arm.open_gripper(degree=15, time_sleep=2)
        pos[2] += VERTICAL_OFFSET
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), SAFE_SPEED)

        # (5) Finally, home the robot, then open the gripper to drop something (ideally).
        arm.home(open_gripper=False)
        arm.open_gripper(degree=90, time_sleep=2)

    # Store the demonstration --- a list of TEN tuples --- if not using the regressor.
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
