"""
I'll try to do some simple motion planning for the set of contours I have to 
see if I can seamlessly get the robot to go where I want it to go. This will
be the open-loop policy baseline (or at least something which I can convert
to be the open-loop...).
"""

import environ
from dvrk.robot import *
from autolab.data_collector import DataCollector
import cv2
import numpy as np
import pickle
import sys
IMDIR = "scripts/images/"

# See `config/daniel_mono_stats.txt`. I think arm1 (respectively, arm2) should
# be equivalent (in theory) for both cameras. There's inevitable noise here.
ARM1_LCAM_HEIGHT = -0.16023154
ARM2_LCAM_HEIGHT = -0.1230049
ARM1_RCAM_HEIGHT = -0.16062703
ARM2_RCAM_HEIGHT = -0.12185628


def initializeRobots():
    d = DataCollector()
    r1 = robot("PSM1") # left (but my right)
    r2 = robot("PSM2") # right (but my left)
    time.sleep(2)
    return (r1,r2,d)


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
    topK = 5
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

    # Show image with contours + exact centers.
    cv2.imshow("Image with topK contours", img_for_drawing)
    key = cv2.waitKey(0) 
    cv2.destroyAllWindows()

    # Manage predictions
    X = np.array(places_to_visit)
    ypred_arm1 = arm1map.predict(X)
    ypred_arm2 = arm2map.predict(X)
    c1 = ARM1_LCAM_HEIGHT if left else ARM1_RCAM_HEIGHT 
    c2 = ARM2_LCAM_HEIGHT if left else ARM2_RCAM_HEIGHT 
    ypred_arm1_full = np.concatenate((ypred_arm1,np.ones((num_points,1))*c1), axis=1)
    ypred_arm2_full = np.concatenate((ypred_arm2,np.ones((num_points,1))*c2), axis=1)
    print("\tHere's where we'll be visiting:\n{}".format(places_to_visit))
    print("\tin matrix form, X.shape is {} and elements are:\n{}".format(X.shape, X))
    print("\tpreds for arm1:\n{}\n\tpreds for arm2:\n{}".format(ypred_arm1, ypred_arm2))
    print("\tfull ypred_arm1:\n{}\n\tfull ypred_arm2:\n{}".format(ypred_arm1_full, ypred_arm2_full))

    print("")
    # Now let's motion plan, with arm1 to start. For angles, I just use something close to the home angle.
    for i,pt_camera in enumerate(places_to_visit):
        arm_pt = ypred_arm1_full[i]
        print("moving arm1 to pixel point {} ...".format(pt_camera))
        post,rott = (tuple(arm_pt), (0.0, 0.0,-160.0))
        pos = [post[0], post[1], post[2]]
        rot = tfx.tb_angles(rott[0], rott[1], rott[2])
        arm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)
        arm1.home()
        arm1.close_gripper()
    arm1.home()

    print("")
    # Then do arm2, with arm2's home angles. Yeah...
    for i,pt_camera in enumerate(places_to_visit):
        arm_pt = ypred_arm2_full[i]
        print("moving arm2 to pixel point {} ...".format(pt_camera))
        post,rott = (tuple(arm_pt), (180.0,-20.0,160.0))
        pos = [post[0], post[1], post[2]]
        rot = tfx.tb_angles(rott[0], rott[1], rott[2])
        arm2.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)
        arm2.home()
        arm2.close_gripper()
    arm2.home()


if __name__ == "__main__":
    arm1, arm2, d = initializeRobots()
    arm1.home()
    arm2.home()

    #cv2.imwrite(IMDIR+"left_camera_image.png",  d.left_image)
    #cv2.imwrite(IMDIR+"right_camera_image.png", d.right_image)
    #cv2.imshow("Left Camera Image", d.left_image)
    #cv2.waitKey(0)
    #cv2.imshow("Right Camera Image", d.right_image)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Load the Random Forest regressors. We saved in tuples, hence load into tuples.
    left_arm1_map,  left_arm2_map  = pickle.load(open('config/daniel_left_mono_model.p'))
    right_arm1_map, right_arm2_map = pickle.load(open('config/daniel_right_mono_model.p'))

    print("\nTesting motion planning using the _left_ camera image.")
    motion_planning(contours_by_size=d.left_contours_by_size, 
                    img=d.left_image, 
                    arm1=arm1,
                    arm2=arm2, 
                    arm1map=left_arm1_map,
                    arm2map=left_arm2_map)
    ##print("\nTesting motion planning using the _right_ camera image.")
    ##motion_planning(contours_by_size=d.right_contours_by_size, 
    ##                img=d.right_image, 
    ##                arm1=arm1,
    ##                arm2=arm2, 
    ##                arm1map=right_arm1_map,
    ##                arm2map=right_arm2_map)
