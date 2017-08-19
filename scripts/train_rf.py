""" 
AFTER we train the rigid body, this will go through the points and apply the predictor.
I'll get systematic errors (if things were working well) so I will MANUALLY adjust the
robot motion. Then train a random forest predictor on that. THEN do again the click-and-crop
thingy that I've been doing.
"""

import cv2
import image_geometry
import numpy as np
import os
import pickle
import sys
from autolab.data_collector import DataCollector
from dvrk.robot import *
np.set_printoptions(suppress=True)

# Double check these as needed. CHECK THE NUMBERS, i.e. v00, v01, etc.
OUTVERSION   = '00' # for _storing_ stuff, use 99 for debugging
VERSION      = '00' # for _loading_ stuff

OUTPUT_FILE  = 'config/calibration_results/data_for_rf_v'+OUTVERSION+'.p'
ESC_KEYS     = [27, 1048603]
ARM1_ZOFFSET = 0.000  # Add 1mm to avoid repeatedly damaging the paper.
C_LEFT_INFO  = pickle.load(open('config/camera_info_matrices/left.p',  'r'))
C_RIGHT_INFO = pickle.load(open('config/camera_info_matrices/right.p', 'r'))
MAX_NUM_ADD  = 36

# OK ... I'm going to have to figure out a better way to deal with rotation. For now
# I will assume taking the average is close enough, but we should check. The issue is 
# that the data from calibration shows systematic trends, e.g. increasing roll. One
# option is to redo calibration, and after I push the end effector, force it to rotate.

lll = pickle.load(open('config/calib_circlegrid_left_v'+VERSION+'_ONELIST.p', 'r'))
rrr = pickle.load(open('config/calib_circlegrid_right_v'+VERSION+'_ONELIST.p', 'r'))
rotations_l = [aa[1] for aa in lll]
rotations_r = [aa[1] for aa in rrr]
rotations_all = np.array(rotations_l + rotations_r)
ROTATION = np.mean(rotations_all, axis=0)
assert ROTATION.shape == (3,)
print("ROTATION: {}".format(ROTATION))

## NOW move on to methods ...

def initializeRobots():
    d = DataCollector()
    r1 = robot("PSM1") # left (but my right)
    r2 = robot("PSM2") # right (but my left)
    time.sleep(2)
    return (r1,r2,d)


def storeData(filename, arm1):
    """ Stores data by repeatedly appending data points to this file (not 
    overriding). Then other code can simply enumerate over the whole thing.
    """
    f = open(filename, 'a')
    pickle.dump(arm1, f)
    f.close()


def left_pixel_to_robot_prediction(left_pt, params):
    """ Given pixels from the left camera (cx,cy) representing camera point, 
    determine the corresponding (x,y,z) robot frame.

    Parameters
    ----------
    left_pt: [(int,int)]
        A tuple of integers representing pixel values from the _left_ camera.
    params: [Dict]
        Dictionary of parameters from `mapping.py`.

    Returns
    -------
    A list of 3 elements representing the predicted robot frame. We WILL apply
    the z-offset here for safety reasons.
    """
    leftx, lefty = left_pt
    left_pt_hom = np.array([leftx, lefty, 1.])
    right_pt = left_pt_hom.dot(params['theta_l2r'])

    # Copy the code I wrote to convert these pts to camera points.
    disparity = np.linalg.norm(left_pt-right_pt)
    stereoModel = image_geometry.StereoCameraModel()
    stereoModel.fromCameraInfo(C_LEFT_INFO, C_RIGHT_INFO)
    (xx,yy,zz) = stereoModel.projectPixelTo3d( (leftx,lefty), disparity )
    camera_pt = np.array([xx, yy, zz])

    # Now I can apply the rigid body and RF (if desired).
    camera_pt = np.concatenate( (camera_pt, np.ones(1)) )
    robot_pt = (params['RB_matrix']).dot(camera_pt)

    # Finally, apply the z-offset. You didn't forget that, did you?
    target = [robot_pt[0], robot_pt[1], robot_pt[2] + ARM1_ZOFFSET]
    if target[2] < -0.18:
        print("Warning! Unsafe target: {}".format(target))
        sys.exit()
    return target


def filter_point(x,y):
    ignore = False
    if (x < 500 or x > 1500 or y < 50 or y > 1000):
        ignore = True
    return ignore


if __name__ == "__main__":
    arm, _, d = initializeRobots()
    print("current arm position: {}".format(arm.get_current_cartesian_position()))
    arm.home()
    arm.close_gripper()
    print("current arm position: {}".format(arm.get_current_cartesian_position()))

    # I will be IGNORING the random forest here, because it doesn't really work.
    params = pickle.load(open('config/mapping_results/params_matrices_v'+VERSION+'.p', 'r'))

    # Use the d.left_image for calibration.
    cv2.imwrite("images/left_image.jpg", d.left_image)
    image_original = cv2.imread("images/left_image.jpg")
    num_added = 0

    # Iterate through valid contours from the _left_ camera (we'll simulate right camera).
    for i, (cX, cY, approx, peri) in enumerate(d.left_contours):  
        if filter_point(cX,cY):
            continue
        if num_added == MAX_NUM_ADD:
            break

        image = image_original.copy()
        cv2.circle(img=image, center=(cX,cY), radius=50, color=(0,0,255), thickness=1)
        cv2.circle(img=image, center=(cX,cY), radius=4, color=(0,0,255), thickness=-1)
        cv2.drawContours(image=image, contours=[approx], contourIdx=-1, color=(0,255,0), thickness=3)
        cv2.imshow("Press ESC if this isn't a desired contour (or a duplicate), press any other key to proceed w/robot movement. Note that num_added = {}".format(num_added), image)
        firstkey = cv2.waitKey(0) 

        if firstkey not in ESC_KEYS:
            # First, determine where the robot will move to based on the pixels.
            target = left_pixel_to_robot_prediction((cX,cY), params)
            pos = [target[0], target[1], target[2]]
            rot = tfx.tb_angles(ROTATION[0], ROTATION[1], ROTATION[2])

            # Robot moves to that point and THEN lowers itself. Will likely be off. 
            # I think 6 seconds is enough for the camera to refresh. But save the frame!
            arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)

            # Eh I decided not to do this. The z-axis is already quite accurate anyway.
            #lowered_pos = [pos[0], pos[1], pos[2] - ARM1_ZOFFSET]
            #arm.move_cartesian_frame_linear_interpolation(tfx.pose(lowered_pos, rot), 0.01)
            time.sleep(5)
            predicted_pos = arm.get_current_cartesian_position()

            # IMPORTANT! Now I do some human motion!!
            updated_image_copy = (d.left_image).copy()
            window_name = "Tried to move to ({},{}), position {}. ADJUST IT AS NEEDED then press any key other than ESC! Or ESC if I made a mistake and need to skip this".format(cX,cY,predicted_pos)
            cv2.imshow(window_name, updated_image_copy)
            key = cv2.waitKey(0)
            if key in ESC_KEYS:
                continue
            cv2.destroyAllWindows()

            # Now record the new position of the arm, and save it.
            new_pos = arm.get_current_cartesian_position()
            data_pt = {'predicted_pos': predicted_pos, 'new_pos': new_pos}
            storeData(OUTPUT_FILE, data_pt)

            # Some stats for debugging, etc.
            num_added += 1
            print("contour {}, data_pt: {} (# added: {})".format(i, data_pt, num_added))
        else:
            print("(not storing contour {})".format(i))

        arm.home()
        arm.close_gripper()
        cv2.destroyAllWindows()
