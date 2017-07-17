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


def save_images(d):
    """ For debugging. """
    cv2.imwrite(IMDIR+"left_camera.png",  d.left_image)
    cv2.imwrite(IMDIR+"right_camera.png", d.right_image)
    cv2.imwrite(IMDIR+"left_camera_proc.png",  d.left_image_proc)
    cv2.imwrite(IMDIR+"right_camera_proc.png", d.right_image_proc)
    cv2.imwrite(IMDIR+"left_camera_gray.png",  d.left_image_gray)
    cv2.imwrite(IMDIR+"right_camera_gray.png", d.right_image_gray)
    cv2.imwrite(IMDIR+"left_camera_circles.png",  d.left_image_circles)
    cv2.imwrite(IMDIR+"right_camera_circles.png", d.right_image_circles)


def initializeRobots(sleep_time=5, wait_for_circles=False):
    """ Initialization w/circles can take a while so let's add a check. """
    d = DataCollector()
    r1 = robot("PSM1") # left (but my right)
    r2 = robot("PSM2") # right (but my left)
    time.sleep(sleep_time)
    if wait_for_circles:
        while (d.left_image_circles is None) or (d.right_image_circles is None):
            time.sleep(1)
        print("Finally finished setting the circles!")
        print("left circles:\n{}".format(d.left_circles))
        print("right circles:\n{}".format(d.right_circles))
    return (r1,r2,d)


def motion_one_arm(places_to_visit, arm, ypred_arm_full, rotation):
    """ Motion plan w/one arm (wrt one camera). For angles, use the `home` angle. """
    print("")
    for i,pt_camera in enumerate(places_to_visit):
        arm_pt = ypred_arm_full[i]
        print("moving arm1 to pixel point {} ...".format(pt_camera))
        post, rott = (tuple(arm_pt), rotation)
        pos = [post[0], post[1], post[2]]
        rot = tfx.tb_angles(rott[0], rott[1], rott[2])
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)
        arm.home()
        arm.close_gripper()
    arm.home()


def motion_planning(contours_by_size, img, arm1, arm2, arm1map, arm2map, left=True):
    """ Simple motion planning. Going from point A to point B, basically.
    
    Parameters
    ----------
    arm1map: [RandomForestRegressor]
        Maps from {left,right} camera's (cX,cY) to arm1's position, assuming fixed height.
    arm2map: [RandomForestRegressor]
        Maps from {left,right} camera's (cX,cY) to arm2's position, assuming fixed height.
    left: [Boolean]
        True if this assumes the left camera, false if right camera.
    """
    topK = 10
    arm1.home()
    arm2.home()
    print("(after calling `home`) psm1 current position: {}".format(arm1.get_current_cartesian_position()))
    print("(after calling `home`) psm2 current position: {}".format(arm2.get_current_cartesian_position()))
    print("We identified {} contours but will keep top {}.".format(len(contours_by_size), topK))
    img_for_drawing = img.copy()
    contours = contours_by_size[:topK]
    cv2.drawContours(img_for_drawing, contours, -1, (0,255,0), 3)
    places_to_visit = []

    # Iterate and find centers. We'll make the robot move to these centers in a sequence.
    for i,cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        # Sometimes I see duplicates, so just handle this case here.
        if (cX,cY) not in places_to_visit:
            cv2.circle(img=img_for_drawing, center=(cX,cY), radius=5, color=(255,0,0), thickness=4)
            places_to_visit.append((cX,cY))
    num_points = len(places_to_visit)

    # Insert any special ordering preferences here and number them in the image.
    # places_to_visit = ... shuffle(places_to_visit) ... ???
    for i,(cX,cY) in enumerate(places_to_visit):
        cv2.putText(img=img_for_drawing, text=str(i), org=(cX,cY), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(0,0,0), thickness=2)

    # Show image with contours + exact centers. Exit if it's not looking good.
    cv2.imshow("Image with topK contours", img_for_drawing)
    key = cv2.waitKey(0) 
    if key == ESC_KEY:
        print("Pressed ESC key. Terminating program...")
        return
    cv2.destroyAllWindows()

    # Manage predictions, store in `ypred_arm{1,2}_full`.
    X = np.array(places_to_visit)
    ypred_arm1 = arm1map.predict(X)
    ypred_arm2 = arm2map.predict(X)
    c1 = ARM1_LCAM_HEIGHT if left else ARM1_RCAM_HEIGHT 
    c2 = ARM2_LCAM_HEIGHT if left else ARM2_RCAM_HEIGHT 
    ypred_arm1_full = np.concatenate((ypred_arm1,np.ones((num_points,1))*c1), axis=1)
    ypred_arm2_full = np.concatenate((ypred_arm2,np.ones((num_points,1))*c2), axis=1)
    print("\tHere's where we'll be visiting:\n{}".format(places_to_visit))
    print("\tin matrix form, X.shape is {} and elements are:\n{}".format(X.shape, X))
    print("\tfull ypred_arm1:\n{}\n\tfull ypred_arm2:\n{}".format(ypred_arm1_full, ypred_arm2_full))

    # Finally, perform motion planning with the predicted places to visit.
    motion_one_arm(places_to_visit, arm1, ypred_arm1_full, rotation=(  0.0,   0.0, -160.0))
    motion_one_arm(places_to_visit, arm2, ypred_arm2_full, rotation=(180.0, -20.0,  160.0))


if __name__ == "__main__":
    arm1, arm2, d = initializeRobots(sleep_time=10, wait_for_circles=True)
    arm1.home()
    arm2.home()

    # Keep these lines for debugging and so forth.
    save_images(d)
    cv2.imshow("Left Image Circles", d.left_image_circles)
    cv2.waitKey(0)
    cv2.imshow("Right Image Circles", d.right_image_circles)
    cv2.waitKey(0)
    print("len(left_circles): {}, len(right_circles): {}".format(len(d.left_circles), len(d.right_circles)))

    # Load the Random Forest regressors. We saved in tuples, hence load into tuples.
    left_arm1_map,  left_arm2_map  = pickle.load(open('config/daniel_left_mono_model_v02_and_v03.p'))
    right_arm1_map, right_arm2_map = pickle.load(open('config/daniel_right_mono_model_v02_and_v03.p'))

    print("\nRunning the OPEN LOOP POLICY using the *left* camera image.")
    motion_planning(contours_by_size=d.left_contours_by_size, 
                    img=d.left_image, 
                    arm1=arm1,
                    arm2=arm2, 
                    arm1map=left_arm1_map,
                    arm2map=left_arm2_map)
    ##print("\nRunning the OPEN LOOP POLICY using the *right* camera image.")
    ##motion_planning(contours_by_size=d.right_contours_by_size, 
    ##                img=d.right_image, 
    ##                arm1=arm1,
    ##                arm2=arm2, 
    ##                arm1map=right_arm1_map,
    ##                arm2map=right_arm2_map)
