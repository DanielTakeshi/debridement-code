""" 
Because I need to detect mouse clicks so that I can find and visualize where the rigid body
and random forests are going wrong. AFTER calibration, I will send the robot end effector to 
many points. Then, once it goes to each point, this code will stop, show an _image_ of the 
target point and the actual robot points. Then I'll drag a box around where the end-effector 
is, and then the code should automatically record stuff. Be careful to record the right stuff,
so it's manual but I'm usually good with this.
"""

import cv2
import image_geometry
import numpy as np
import os
import pickle
import sys
from autolab.data_collector import DataCollector
from dvrk.robot import *

# Double check these as needed. CHECK THE NUMBERS, i.e. v00, v01, etc.
VERSION      = '00'
ESC_KEYS     = [27, 1048603]
ARM1_ZOFFSET = 0.002  # Add 2mm to avoid repeatedly damaging the paper.
ROTATION     = (0.0, 0.0, -160.0)
OUTPUT_FILE  = 'config/calibration_results/data_v'+VERSION+'.p'
IMDIR        = 'images/check_regressors_v'+VERSION+'/'
USE_RF       = False
C_LEFT_INFO  = pickle.load(open('config/camera_info_matrices/left.p',  'r'))
C_RIGHT_INFO = pickle.load(open('config/camera_info_matrices/right.p', 'r'))

# Initialize the list of reference points 
POINTS          = []
CENTER_OF_BOXES = []


def initializeRobots():
    d = DataCollector()
    r1 = robot("PSM1") # left (but my right)
    r2 = robot("PSM2") # right (but my left)
    time.sleep(2)
    return (r1,r2,d)


def storeData(filename, arm1):
    """ Stores data by repeatedly appending data points to this file.
    Then other code can simply enumerate over the whole thing.
    """
    f = open(filename, 'a')
    pickle.dump(arm1, f)
    f.close()


def left_pixel_to_robot_prediction(camera_pt, params):
    """ Given pixels from the left camera (cx,cy) representing camera point, 
    determine the corresponding (x,y,z) robot frame.

    Parameters
    ----------
    camera_pt: [(int,int)]
        A tuple of integers representing pixel values from the _left_ camera.
    params: [Dict]
        Dictionary of parameters from `mapping.py`.

    Returns
    -------
    A list of 3 elements representing the predicted robot frame. We WILL apply
    the z-offset here for safety reasons.
    """
    leftx, lefty = camera_pt
    left_pt_hom = np.array([leftx, lefty, 1.])
    right_pt = left_pt_hom.dot(params['theta_l2r'])

    # Copy the code I wrote to convert these pts to camera points.
    disparity = np.linalg.norm(camera_pt-right_pt)
    stereoModel = image_geometry.StereoCameraModel()
    stereoModel.fromCameraInfo(C_LEFT_INFO, C_RIGHT_INFO)
    (xx,yy,zz) = stereoModel.projectPixelTo3d( (leftx,lefty), disparity )
    camera_pt = np.array([xx, yy, zz])

    # Now I can apply the rigid body and RF (if desired).
    camera_pt = np.concatenate(camera_pt, np.zeros((1,1)))
    robot_pt = (params['RB_matrix']).dot(camera_pt)
    if USE_RF:
        residuals = params['rf'].predict(camera_pt)
        robot_pt = robot_pt - residuals

    # Finally, apply the z-offset. You didn't forget that, did you?
    target = [robot_pt[0], robot_pt[1], robot_pt[2] + ARM1_ZOFFSET]
    return target


def click_and_crop(event, x, y, flags, param):
    global POINTS, CENTER_OF_BOXES
             
    # If left mouse button clicked, record the starting (x,y) coordinates 
    # and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS.append((x,y))
                                                 
    # Check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # Record ending (x,y) coordinates and indicate that cropping is finished AND save center!
        POINTS.append((x,y))

        upper_left = POINTS[-2]
        lower_right = POINTS[-1]
        assert upper_left[0] < lower_right[0]
        assert upper_left[1] < lower_right[1]
        center_x = int(upper_left[0] + (lower_right[0]-upper_left[0])/2)
        center_y = int(upper_left[1] + (lower_right[1]-upper_left[1])/2)
        CENTER_OF_BOXES.append( (center_x,center_y) )
        
        # Draw a rectangle around the region of interest, w/center point. Blue=Before, Red=AfteR.
        cv2.rectangle(img=updated_image_copy, 
                      pt1=POINTS[-2], 
                      pt2=POINTS[-1], 
                      color=(0,0,255), 
                      thickness=2)
        cv2.circle(img=updated_image_copy, 
                   center=CENTER_OF_BOXES[-1], 
                   radius=6, 
                   color=(0,0,255), 
                   thickness=-1)
        cv2.putText(img=updated_image_copy, 
                    text="{}".format(CENTER_OF_BOXES[-1]), 
                    org=CENTER_OF_BOXES[-1],  
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, 
                    color=(0,0,255), 
                    thickness=2)
        cv2.imshow("This is in the click and crop method AFTER the rectangle. "+
                   "(Press any key, or ESC If I made a mistake.)", updated_image_copy)


if __name__ == "__main__":
    arm, _, d = initializeRobots()
    arm.close_gripper()
    arm.home()
    params = pickle.load(open('config/mapping_results/params_matrices_v'+VERSION+'.p', 'r'))

    # Use the d.left_image for calibration. Originally I used a saved image, but it should
    # be determined here since the paper location and camera might adjust slightly.
    cv2.imwrite("images/left_image.jpg", d.left_image)
    image_original = cv2.imread("images/left_image.jpg")
    num_added = 0

    # Iterate through valid contours from the _left_ camera (we'll simulate right camera).
    for i, (cX, cY, approx, peri) in enumerate(d.left_contours):  
        image = image_original.copy()
        cv2.circle(img=image, center=(cX,cY), radius=50, color=(0,0,255), thickness=1)
        cv2.drawContours(image=image, contours=[approx], contourIdx=-1, color=(0,255,0), thickness=3)
        cv2.imshow("Press ESC if this isn't a desired contour (or a duplicate), "+
                    "press any other key to proceed w/robot movement.", image)
        firstkey = cv2.waitKey(0) 

        if firstkey not in ESC_KEYS:
            # First, determine where the robot will move to based on the pixels.
            target = left_pixel_to_robot_prediction((cX,cY), params):
            pos = [target[0], target[1], target[2]]
            rot = tfx.tb_angles(ROTATION[0], ROTATION[1], ROTATION[2])

            # Robot moves to that point and will likely be off. 
            # I think 6 seconds is enough for the camera to refresh.
            arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)
            time.sleep(6)

            # Update image and put center coordinate there. Blue=Before, Red=AfteR.
            updated_image_copy = (d.left_image).copy()
            cv2.circle(img=updated_image_copy, center=(cX,cY), radius=6, color=(255,0,0), thickness=-1)
            cv2.putText(img=updated_image_copy, 
                        text="{}".format((cX,cY)), 
                        org=(cX,cY), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=1, 
                        color=(255,0,0), 
                        thickness=2)

            # Now we apply the callback and drag a box around the end-effector on the (updated!) image.
            window_name = "Robot has tried to move to ({},{}). Click and drag a box "
                "around its end effector. Then press any key (or ESC if I made mistake).".format(cX,cY)
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, click_and_crop)
            cv2.imshow(window_name, updated_image_copy)
            key = cv2.waitKey(0)
            if key in ESC_KEYS:
                continue

            # Now save the image with the next available index. It will contain the contours.
            index = len(os.listdir(IMDIR))
            cv2.imwrite(IMDIR+"/point_"+str(index).zfill(2)+".png", updated_image_copy)
            cv2.destroyAllWindows()
 
            # Get position and orientation of the arm, save (in a dictionary), & reset. 
            # I.e. the target_pos is what the random forest predicted for the target position.
            frame = arm.get_current_cartesian_position()
            new_pos = tuple(frame.position[:3])
            new_rot = tfx.tb_angles(frame.rotation)
            new_rot = (new_rot.yaw_deg, new_rot.pitch_deg, new_rot.roll_deg)
            data_pt = {'target_pos': pos,
                       'target_rot': ROTATION,
                       'actual_pos': new_pos,
                       'actual_rot': new_rot,
                       'center_target_pixels': (cX,cY),
                       'center_actual_pixels': CENTER_OF_BOXES[-1]}
            storeData(OUTPUT_FILE, data_pt)

            # Some stats for debugging, etc.
            num_added += 1
            print("contour {}, data_pt: {} (# added: {})".format(i, data_pt, num_added))
            assert (2*num_added) == (2*len(CENTER_OF_BOXES)) == len(POINTS)

        else:
            print("(not storing contour {})".format(i))

        arm.home()
        arm.close_gripper()
        cv2.destroyAllWindows()
