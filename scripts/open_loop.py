"""
This is the open-loop policy. We will do this as a baseline policy. 
This must be applied on our actual experimental setup.
"""

import environ
from dvrk.robot import *
from autolab.data_collector import DataCollector
import cv2
import numpy as np
import pickle
import sys
IMDIR = "scripts/images/"
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


def save_images(d):
    """ For debugging/visualization. """
    cv2.imwrite(IMDIR+"left_proc.png",  d.left_image_proc)
    cv2.imwrite(IMDIR+"left_gray.png",  d.left_image_gray)
    cv2.imwrite(IMDIR+"right_proc.png", d.right_image_proc)
    cv2.imwrite(IMDIR+"right_gray.png", d.right_image_gray)


def call_wait_key(nothing):
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


def motion_one_arm(places_to_visit, arm, ypred_arm_full, rotation):
    """ Motion plan w/one arm (wrt one camera). For angles, use the `home` angle. 

    Note that we can't just call open gripper and close gripper and expect that the robot will
    be finished with its motion. The way the dvrk works is that commands are called sequentially
    immediately, even if their actual impact isn't felt in "dvrk-space", which is why I need a
    bunch of time.sleep() calls. I'm not sure if there's a better way around this.
    
    places_to_visit: [list] 
        List of tuples (cX,cY) indicating the pixels to which the arm should move.
    arm: [dvrk arm] 
        Either the left or right DVRK arm, from calling `robot("PSM{1,2}")`.
    ypred_arm_full: [np.array]
        Numpy array of shape (N,3) where N is the number of points to visit and each row
        consists of the x and y coordinates (in pixel space) along with a fixed height.
    rotation: [tuple] 
        Completes the specification of the points to move to. These are known ahead of time
        for both arms.
    """
    print("")
    assert ypred_arm_full.shape[0] == len(places_to_visit)
    arm.open_gripper(degree=90, time_sleep=2)

    for i,pt_camera in enumerate(places_to_visit):
        arm_pt = ypred_arm_full[i]

        # First, get the arm to (the rough general area of) the target.
        print("moving arm1 to pixel point {} ...".format(pt_camera))
        post, rott = (tuple(arm_pt), rotation)
        pos = [post[0], post[1], post[2]]
        rot = tfx.tb_angles(rott[0], rott[1], rott[2])
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)

        # Move gripper *downwards* to target object (ideally).
        pos[2] -= 0.01
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), SAFE_SPEED)
   
        # Close gripper and move *upwards*, ideally grabbing the object..
        #arm.close_gripper(time_sleep=0)
        arm.open_gripper(degree=10, time_sleep=2)
        pos[2] += 0.01
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), SAFE_SPEED)
 
        # Finally, home the robot, then open the gripper to drop something (ideally).
        arm.home(open_gripper=False)
        arm.open_gripper(degree=90, time_sleep=2)
    arm.home()


def motion_planning(contours_by_size, circles, use_contours, img, arm1, arm2, arm1map, arm2map, left=True):
    """ Simple motion planning. Going from point A to point B, basically.
    
    Parameters
    ----------
    contours_by_size:
    circles:
    use_contours:
    img:
    arm1:
    arm2:
    arm1map: [RandomForestRegressor]
        Maps from {left,right} camera's (cX,cY) to arm1's position, assuming fixed height.
    arm2map: [RandomForestRegressor]
        Maps from {left,right} camera's (cX,cY) to arm2's position, assuming fixed height.
    left: [Boolean]
        True if this assumes the left camera, false if right camera.
    """
    topK = TOPK
    arm1.home()
    arm2.home()
    print("Note: use_contours: {}".format(use_contours))
    #print("(after calling `home`) psm1 current position: {}".format(arm1.get_current_cartesian_position()))
    #print("(after calling `home`) psm2 current position: {}".format(arm2.get_current_cartesian_position()))
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

    places_to_visit = places_to_visit[:topK]
    num_points = len(places_to_visit)

    # Insert any special ordering preferences here and number them in the image.
    # places_to_visit = ... shuffle(places_to_visit) ... ???
    for i,(cX,cY) in enumerate(places_to_visit):
        cv2.putText(img=img_for_drawing, text=str(i), org=(cX,cY), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(0,0,0), thickness=2)
        #print(i,cX,cY)
        cv2.circle(img=img_for_drawing, center=(cX,cY), radius=3, color=(0,0,255), thickness=-1)


    # Show image with contours + exact centers. Exit if it's not looking good.
    cv2.imshow("Image with topK contours (exit if not looking good)", img_for_drawing)
    key = cv2.waitKey(0) 
    cv2.destroyAllWindows()
    if key == ESC_KEY:
        print("Pressed ESC key. Terminating program...")
        return

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

    # Finally, perform motion planning with the predicted places to visit.
    motion_one_arm(places_to_visit, arm1, ypred_arm1_full, rotation=(  0.0,   0.0, -160.0))
    #motion_one_arm(places_to_visit, arm2, ypred_arm2_full, rotation=(180.0, -20.0,  160.0))


if __name__ == "__main__":
    """ See the top of the file for program-wide arguments. """
    arm1, arm2, d = initializeRobots(sleep_time=4)
    arm1.home()
    arm2.home()
    
    # Keep these lines for debugging and so forth.
    #save_images(d)
    show_images(d)

    # Load the Random Forest regressors. We saved in tuples, hence load into tuples.
    left_arm1_map,  left_arm2_map  = pickle.load(open('config/daniel_left_mono_model_v02_and_v03.p'))
    right_arm1_map, right_arm2_map = pickle.load(open('config/daniel_right_mono_model_v02_and_v03.p'))

    if DO_LEFT_CAMERA:
        print("\nRunning the OPEN LOOP POLICY using the *left* camera image.")
        motion_planning(contours_by_size=d.left_contours_by_size, 
                        circles=d.left_circles,
                        use_contours=True,
                        img=d.left_image, 
                        arm1=arm1,
                        arm2=arm2, 
                        arm1map=left_arm1_map,
                        arm2map=left_arm2_map,
                        left=True)

    if DO_RIGHT_CAMERA:
        print("\nRunning the OPEN LOOP POLICY using the *right* camera image.")
        motion_planning(contours_by_size=d.right_contours_by_size, 
                        circles=d.right_circles,
                        use_contours=True,
                        img=d.right_image, 
                        arm1=arm1,
                        arm2=arm2, 
                        arm1map=right_arm1_map,
                        arm2map=right_arm2_map,
                        left=False)
