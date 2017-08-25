""" For reducing clutter. """

from autolab.data_collector import DataCollector
from dvrk.robot import *
import cv2
import image_geometry
import numpy as np
import os
import pickle
import sys
np.set_printoptions(suppress=True)

# Because I had this almost everywhere!
ESC_KEYS = [27, 1048603]


def initializeRobots(sleep_time=2):
    """ Because I was using this in almost every script! """
    d = DataCollector()
    r1 = robot("PSM1") # left (but my right)
    r2 = robot("PSM2") # right (but my left)
    time.sleep(sleep_time)
    return (r1,r2,d)


def storeData(filename, arm1):
    """ 
    Stores data by repeatedly appending data points to this file (not overriding). 
    Then other code can simply enumerate over the whole thing.
    """
    f = open(filename, 'a')
    pickle.dump(arm1, f)
    f.close()


def get_num_stuff_in_pickle_file(filename):
    """ Counting stuff in a pickle file! """
    f = open(filename,'r')
    num = 0
    while True:
        try:
            d = pickle.load(f)
            num += 1
        except EOFError:
            break
    return num


def call_wait_key(nothing=None):
    """ I have an ESC which helps me exit program. """
    ESC_KEYS = [27, 1048603]
    key = cv2.waitKey(0)
    if key in ESC_KEYS:
        print("Pressed ESC key. Terminating program...")
        sys.exit()


## def save_images(d):
##     """ For debugging/visualization of DataCollector. """
##     cv2.imwrite(IMAGE_DIR+"left_proc.png",  d.left_image_proc)
##     cv2.imwrite(IMAGE_DIR+"left_gray.png",  d.left_image_gray)
##     #cv2.imwrite(IMAGE_DIR+"right_proc.png", d.right_image_proc)
##     #cv2.imwrite(IMAGE_DIR+"right_gray.png", d.right_image_gray)


def show_images(d):
    """ For debugging/visualization of DataCollector. """
    #call_wait_key(cv2.imshow("Left Processed", d.left_image_proc))
    #call_wait_key(cv2.imshow("Left Gray",      d.left_image_gray))
    call_wait_key(cv2.imshow("Left BoundBox",  d.left_image_bbox))
    #call_wait_key(cv2.imshow("Left Circles",   d.left_image_circles))
    #print("Circles (left):\n{}".format(d.left_circles))


def left_pixel_to_robot_prediction(left_pt, params, better_rf, 
        ARM1_XOFFSET, ARM1_YOFFSET, ARM1_ZOFFSET, USE_RF=True, bad_rf=False):
    """ Given pixels from the left camera (cx,cy) representing camera point, 
    determine the corresponding (x,y,z) robot frame.

    Parameters
    ----------
    left_pt: [(int,int)]
        A _single_ tuple of integers representing pixel values from the _left_ camera.
        (Sorry, the way I list parameters makes it look like this is a list... it's not.)
    params: [Dict]
        Dictionary of parameters from `mapping.py`.
    better_rf: [Random Forest]
        Used for the random forest. Ignore anything R.F.-related in `params`!
    ARM1_{X,Y,Z}OFFSET: [float]
        These are floats used for making offsets.
    USE_RF: [boolean]
        Whether to apply the random forest or not (we usually should).
    bad_rf: [boolean]
        If True, apply the RF that was trained with the rigid body (i.e. not the human-
        guided version). Should normally be `False`.

    Returns
    -------
    A list of 3 elements representing the predicted robot frame. We WILL apply
    the z-offset here for safety reasons.
    """
    C_LEFT_INFO  = pickle.load(open('config/camera_info_matrices/left.p',  'r'))
    C_RIGHT_INFO = pickle.load(open('config/camera_info_matrices/right.p', 'r'))

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

    if USE_RF:
        if bad_rf:
            residuals = np.squeeze( params['rf_residuals'].predict([camera_pt[:3]]) )[:2] 
        else:
            residuals = np.squeeze( better_rf.predict([robot_pt]) ) # Use _robot_point_ !!
        assert len(residuals) == 2
        residuals = np.concatenate((residuals,np.zeros(1))) # Add zero for z-coord.
        robot_pt = robot_pt - residuals
        print("residuals:             {}".format(residuals))

    print("left_pt/right_pt:      {},{}".format(left_pt,right_pt))
    print("camera_pt:             {}".format(camera_pt))
    print("(predicted) robot_pt:  {}".format(robot_pt))

    # Finally, apply the offsets. You didn't forget that, did you?
    target = [robot_pt[0]+ARM1_XOFFSET, robot_pt[1]+ARM1_YOFFSET, robot_pt[2]+ARM1_ZOFFSET]
    if target[2] < -0.18:
        print("Warning! Unsafe target: {}".format(target))
        sys.exit()
    return target


def get_average_rotation(version):
    """
    OK ... I'm going to have to figure out a better way to deal with rotation. For now
    I will assume taking the average is close enough, but we should check. The issue is 
    that the data from calibration shows systematic trends, e.g. increasing roll. One
    option is to redo calibration, and after I push the end effector, force it to rotate.

    TODO: Deprecate this method. However, I should use it in case I wnat to inspect the
    average rotation to ensure that I didn't adjust it too much during human calibration.
    """
    print("WARNING: this method will be deprecated!!")
    lll = pickle.load(open('config/calib_circlegrid_left_v'+version+'_ONELIST.p', 'r'))
    rrr = pickle.load(open('config/calib_circlegrid_right_v'+version+'_ONELIST.p', 'r'))
    rotations_l = [aa[1] for aa in lll]
    rotations_r = [aa[1] for aa in rrr]
    rotations_all = np.array(rotations_l + rotations_r)
    ROTATION = np.mean(rotations_all, axis=0)
    assert ROTATION.shape == (3,)
    print("ROTATION: {}".format(ROTATION))
    return ROTATION


def get_rotation_from_version(version):
    """
    Given a version number from `scripts/calibrate_onearm.py`, we figure out the rotation.
    Right now this is a lot of trial and error, but there isn't much of an alternative.
    Replaces the old method of `get_average_rotation`.
    """
    if version == '00' or version == '10':
        return [0, -10, -170]
    elif version == '01':
        return [0, 0, -165]
    else:
        raise ValueError()


def filter_point(x, y, xlower, xupper, ylower, yupper):
    """ 
    Used in case we want to filter out contours that aren't in some area. 
    Returns True if we _should_ ignore the point.
    """
    ignore = False
    if (x < xlower or x > xupper or y < ylower or y > yupper):
        ignore = True
    return ignore


def move(arm, pos, rot, SPEED_CLASS):
    """ Handles the different speeds we're using.
    
    Parameters
    ----------
    arm: [dvrk arm]
        The current DVK arm.
    pos: [list]
        The desired position.
    rot: [list]
        The desired rotation, in list form.
    SPEED_CLASS: [String]
        Slow, Medium, or Fast.
    """
    if SPEED_CLASS == 'Slow':
        arm.move_cartesian_frame_linear_interpolation(
                tfx.pose(pos, tfx.tb_angles(rot[0],rot[1],rot[2])), 0.03
        )
    elif SPEED_CLASS == 'Medium':
        arm.move_cartesian_frame_linear_interpolation(
                tfx.pose(pos, tfx.tb_angles(rot[0],rot[1],rot[2])), 0.06
        )
    elif SPEED_CLASS == 'Fast':
        arm.move_cartesian_frame(tfx.pose(pos, tfx.tb_angles(rot[0],rot[1],rot[2])))
    else:
        raise ValueError()
