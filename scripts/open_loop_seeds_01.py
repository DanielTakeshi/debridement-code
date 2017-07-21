"""
This is the open-loop policy. We will do this as a baseline policy. This must be applied on our actual experimental setup.

How about this to start: focus on having four seeds in a row. Program an open-loop policy to pick up seeds in order.
The trajectory is thus split into four timesteps. The times we intervene are for visual servoing, when we have to
figure out where specifically to move. The other parts of the task are a bit irrelevant. This scenario is indexed as "01".
"""

import environ
from dvrk.robot import *
from autolab.data_collector import DataCollector
import cv2
import numpy as np
import pickle
import sys
IMDIR = "scripts/images/"
DEMO_FILE_NAME = 'data/demos_seeds_01.p'
ESC_KEY = 27

# See `config/daniel_mono_stats_v02_and_v03.txt`. I think arm1 (respectively, arm2) 
# should be equivalent (in theory) for both cameras. There's inevitable noise here.
# That the height of our experimental setup is slightly lower, which is fine because
# it means we can move to a point and then gradually lower the end-effectors.
ARM1_LCAM_HEIGHT = -0.16191448
ARM2_LCAM_HEIGHT = -0.12486849
ARM1_RCAM_HEIGHT = -0.16305105
ARM2_RCAM_HEIGHT = -0.12607518

###########################
# ADJUST THESE AS NEEDED! #
###########################
TOPK            = 4 
EXTRA_HEIGHT    = 0.015
WAIT_CIRCLES    = False
DO_LEFT_CAMERA  = True
DO_RIGHT_CAMERA = False
VERTICAL_OFFSET = 0.012


def save_images(d):
    """ For debugging/visualization. """
    cv2.imwrite(IMDIR+"left_proc.png",  d.left_image_proc)
    cv2.imwrite(IMDIR+"left_gray.png",  d.left_image_gray)
    cv2.imwrite(IMDIR+"right_proc.png", d.right_image_proc)
    cv2.imwrite(IMDIR+"right_gray.png", d.right_image_gray)


def call_wait_key(nothing=None):
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

    #call_wait_key(cv2.imshow("Right Processed", d.right_image_proc))
    #call_wait_key(cv2.imshow("Right Gray",      d.right_image_gray))
    call_wait_key(cv2.imshow("Right BoundBox",  d.right_image_bbox))
    #call_wait_key(cv2.imshow("Right Circles",   d.right_image_circles))

    print("Circles (left):\n{}".format(d.left_circles))
    print("Circles (right):\n{}".format(d.right_circles))


def initializeRobots(sleep_time=5):
    d = DataCollector()
    r1 = robot("PSM1") # left (but my right)
    r2 = robot("PSM2") # right (but my left)
    time.sleep(sleep_time)
    if WAIT_CIRCLES:
        while (d.left_image_circles is None) or (d.right_image_circles is None):
            print("At least one circle image is None. Waiting ...")
            time.sleep(5)
    return (r1,r2,d)


def store_demonstration_01(places_to_visit, d, arm, ypred_arm_full, rotation, demo_file=None):
    """ This will store a full trajectory for the four seeds cases.

    Note that we can't just call open gripper and close gripper and expect that the robot will
    be finished with its motion. The way the dvrk works is that commands are called sequentially
    immediately, even if their actual impact isn't felt in "dvrk-space", which is why I need a
    bunch of time.sleep() calls. I'm not sure if there's a better way around this.

    ASSUMES THE LEFT ARM ONLY for simplicity. And the left camera!
    
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
    rotation: [tuple] 
        Completes the specification of the points to move to. These are known ahead of time
        for both arms.
    demo_file: [String]
        Path for the pickle file where we store human-guided demonstrations.
    """
    print("")
    assert ypred_arm_full.shape[0] == len(places_to_visit)
    arm.home()
    arm.open_gripper(degree=90, time_sleep=2)

    # This will store our demonstration. Store the starting location w/angles as well!
    demo = [ (arm.get_current_cartesian_position(), 90) ]

    for i,pt_camera in enumerate(places_to_visit):
        arm_pt = ypred_arm_full[i]

        # (1) Get the arm to (the rough general area of) the target.
        print("Moving arm1 to PIXELS point {} ...".format(pt_camera))
        post, rott = (tuple(arm_pt), rotation)
        pos = [post[0], post[1], post[2]]
        rot = tfx.tb_angles(rott[0], rott[1], rott[2])
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)
        frame = arm.get_current_cartesian_position()
        print("appending frame {}".format(frame))
        demo.append( (frame, 90) )

        # (2) IMPORTANT. HUMAN-GUIDED STUFF HAPPENS HERE! (Press ESC to abort.)
        # Before pressing any key, move the arm to the correct location KEEPING height fixed as much as possible.
        # Unfortunately, that's tricky in itself. Not sure easiest way since we don't have an API for that. :-(
        time.sleep(1)
        call_wait_key(cv2.imshow("Left Gray", d.left_image_gray))
        cv2.destroyAllWindows()
        frame = arm.get_current_cartesian_position()
        print("appending frame {}".format(frame))
        demo.append( (frame, 90) )

        # (3) Move gripper *downwards* to target object (ideally), using new frame.
        pos = (frame.position[:3])
        rot = tfx.tb_angles(frame.rotation) # keep its (new) rotation, not the one I sent as the function input
        pos[2] -= VERTICAL_OFFSET
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), SAFE_SPEED)
        frame = arm.get_current_cartesian_position()
        print("appending frame {}".format(frame))
        demo.append( (frame, 90) )

        # (4) Close gripper and move *upwards*, ideally grabbing the object..
        arm.open_gripper(degree=10, time_sleep=2)
        pos[2] += VERTICAL_OFFSET
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), SAFE_SPEED)
        demo.append( (arm.get_current_cartesian_position(), 10) )

        # (5) Finally, home the robot, then open the gripper to drop something (ideally).
        arm.home(open_gripper=False)
        arm.open_gripper(degree=90, time_sleep=2)
        demo.append( (arm.get_current_cartesian_position(), 90) )
    arm.home()

    # Store the demonstration.
    f = open(DEMO_FILE_NAME, 'a')
    pickle.dump(demo, f)
    f.close()


def motion_planning_01(contours_by_size, circles, use_contours, img, arm1, arm2, arm1map, arm2map):
    """ Simple motion planning. Going from point A to point B, basically. This gets everything _set_up_
    for the actual motion planning. This is '01' which refers to the setting where we have four seeds in
    a row and must pick them up. ASSUMES LEFT ARM.
    
    Parameters
    ----------
    contours_by_size:
    circles:
    use_contours: [boolean]
        True if the contours are used for determining targets, False if circles. Use the contours!
    img:
    arm1:
    arm2:
    arm1map: [RandomForestRegressor]
        Maps from {left,right} camera's (cX,cY) to arm1's position, assuming fixed height.
    arm2map: [RandomForestRegressor]
        Maps from {left,right} camera's (cX,cY) to arm2's position, assuming fixed height.
    """
    topK = TOPK
    left=True
    arm1.home()
    arm2.home()
    print("Note: use_contours: {}".format(use_contours))
    print("We identified {} contours but will keep top {}.".format(len(contours_by_size), topK))
    print("We identified {} circles.".format(len(circles)))
    img_for_drawing = img.copy()
    contours = list(contours_by_size)
    if use_contours:
        cv2.drawContours(img_for_drawing, contours, -1, (0,255,0), 3)
    places_to_visit = []

    if use_contours:
        # Iterate and find centers. We'll make the robot move to these centers in a sequence.
        # Note that duplicate contours should be detected beforehand.
        for i,cnt in enumerate(contours):
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            places_to_visit.append((cX,cY))
    else:
        # Only if we're using HoughCircles. They're pretty bad.
        for i,(x,y,r) in enumerate(circles):
            places_to_visit.append((x,y))

    # Collect only topK places to visit and insert ordering preferences. I like going from right to left.
    places_to_visit = places_to_visit[:topK]
    num_points = len(places_to_visit)
    places_to_visit = sorted(places_to_visit, key=lambda x:x[0], reverse=True)

    # Number the places to visit in an image so I see them.
    for i,(cX,cY) in enumerate(places_to_visit):
        cv2.putText(img=img_for_drawing, text=str(i), org=(cX,cY), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(0,0,0), thickness=2)
        #print(i,cX,cY)
        cv2.circle(img=img_for_drawing, center=(cX,cY), radius=3, color=(0,0,255), thickness=-1)

    # Show image with contours + exact centers. Exit if it's not looking good.
    cv2.imshow("Image with topK contours (exit if not looking good)", img_for_drawing)
    call_wait_key()
    cv2.destroyAllWindows()

    # Manage predictions, store in `ypred_arm{1,2}_full`.
    X = np.array(places_to_visit)
    ypred_arm1 = arm1map.predict(X)
    ypred_arm2 = arm2map.predict(X)
    c1 = (ARM1_LCAM_HEIGHT+EXTRA_HEIGHT) if left else (ARM1_RCAM_HEIGHT+EXTRA_HEIGHT)  
    c2 = (ARM2_LCAM_HEIGHT+EXTRA_HEIGHT) if left else (ARM2_RCAM_HEIGHT+EXTRA_HEIGHT) 
    ypred_arm1_full = np.concatenate((ypred_arm1,np.ones((num_points,1))*c1), axis=1)
    ypred_arm2_full = np.concatenate((ypred_arm2,np.ones((num_points,1))*c2), axis=1)
    print("\tHere's where we'll be visiting:\n{}".format(places_to_visit))
    print("\tin matrix form, X.shape is {} and elements are:\n{}".format(X.shape, X))
    print("\tfull ypred_arm1:\n{}\n\tfull ypred_arm2:\n{}".format(ypred_arm1_full, ypred_arm2_full))

    # Perform a human-guided open-loop demonstration.
    store_demonstration_01(places_to_visit, d, arm1, ypred_arm1_full, rotation=(  0.0,   0.0, -160.0))


if __name__ == "__main__":
    """ See the top of the file for program-wide arguments. """
    arm1, arm2, d = initializeRobots(sleep_time=4)
    arm1.home()
    print("arm1 home: {}".format(arm1.get_current_cartesian_position()))
    arm2.home()
    print("arm2 home: {}".format(arm2.get_current_cartesian_position()))
    
    # Keep these lines for debugging and so forth.
    #save_images(d)
    show_images(d)

    # Load the Random Forest regressors. We saved in tuples, hence load into tuples.
    left_arm1_map,  left_arm2_map  = pickle.load(open('config/daniel_left_mono_model_v02_and_v03.p'))
    right_arm1_map, right_arm2_map = pickle.load(open('config/daniel_right_mono_model_v02_and_v03.p'))

    if DO_LEFT_CAMERA:
        print("\nRunning the OPEN LOOP POLICY using the *left* camera image.")
        motion_planning_01(contours_by_size=d.left_contours_by_size, 
                           circles=d.left_circles,
                           use_contours=True,
                           img=d.left_image, 
                           arm1=arm1,
                           arm2=arm2, 
                           arm1map=left_arm1_map,
                           arm2map=left_arm2_map)
    if DO_RIGHT_CAMERA:
        raise NotImplementedError()
