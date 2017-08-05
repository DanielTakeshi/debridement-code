""" 
Because I need to detect mouse clicks so that I can find and visualize where the random forests are going wrong.
The idea is that AFTER calibration, we will send the robot to many points. Then, once it goes to each point, this
code will stop, show an image of the target point and the actual robot points. Then I'll drag a box around where
the end-effector is, and then the code should automatically record stuff.
"""

import cv2
import numpy as np
import os
import pickle
import sys
from autolab.data_collector import DataCollector
from dvrk.robot import *

# Double check these as needed.
IMDIR           = 'images/check_regressors/'
ORIGINAL_IMAGE  = 'images/check_calibration/calibration_blank_image.jpg'
OUTPUT_FILE     = 'visuals_calibration/data_v00.p'
RF_REGRESSOR    = 'config/daniel_final_mono_map_01.p'
ESC_KEYS        = [27, 1048603] # IDK why I need 1048603 ...
ARM1_ZCOORD     = -0.16873688
ROTATION        = (0.0, 0.0, -160.0)

# initialize the list of reference points 
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


def click_and_crop(event, x, y, flags, param):
    global POINTS, CENTER_OF_BOXES
             
    # If left mouse button clicked, record the starting (x,y) coordinates and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS.append((x,y))
                                                 
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x,y) coordinates and indicate that the cropping operation is finished AND save center!
        POINTS.append((x,y))

        upper_left = POINTS[-2]
        lower_right = POINTS[-1]
        assert upper_left[0] < lower_right[0]
        assert upper_left[1] < lower_right[1]
        center_x = int(upper_left[0] + (lower_right[0]-upper_left[0])/2)
        center_y = int(upper_left[1] + (lower_right[1]-upper_left[1])/2)
        CENTER_OF_BOXES.append( (center_x,center_y) )
        
        # Draw a rectangle around the region of interest, along with the center point. Blue=Before, Red=AfteR.
        cv2.rectangle(img=updated_image_copy, pt1=POINTS[-2], pt2=POINTS[-1], color=(0,0,255), thickness=2)
        cv2.circle(img=updated_image_copy, center=CENTER_OF_BOXES[-1], radius=6, color=(0,0,255), thickness=-1)
        cv2.putText(img=updated_image_copy, text="{}".format(CENTER_OF_BOXES[-1]), org=CENTER_OF_BOXES[-1], 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)
        cv2.imshow("This is in the click and crop method AFTER the rectangle. (Press any key.)", updated_image_copy)


if __name__ == "__main__":
    arm, _, d = initializeRobots()
    arm.close_gripper()
    arm.home()
    arm_map = pickle.load(open(RF_REGRESSOR))

    # Load the original image used for calibration and iterate through the contours, checking them if valid.
    image_original = cv2.imread(ORIGINAL_IMAGE)
    num_added = 0

    for i, (cX, cY, approx, peri) in enumerate(d.left_contours):  
        image = image_original.copy()
        cv2.circle(img=image, center=(cX,cY), radius=50, color=(0,0,255), thickness=1)
        cv2.drawContours(image=image, contours=[approx], contourIdx=-1, color=(0,255,0), thickness=3)
        cv2.imshow("Press ESC if this isn't a desired contour (or a duplicate), press any other key to proceed w/robot movement.", image)
        firstkey = cv2.waitKey(0) 

        if firstkey not in ESC_KEYS:
            # First, determine where the robot will move to based on the pixels.
            target = np.squeeze( arm_map.predict([[cX,cY]]) )
            pos = [target[0], target[1], ARM1_ZCOORD]
            rot = tfx.tb_angles(ROTATION[0], ROTATION[1], ROTATION[2])

            # Robot moves to that point and will likely be off. I think 6 seconds is enough for the camera to refresh.
            arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)
            time.sleep(6)

            # Get the updated image and put the center coordinate there (make it clear). Blue=Before, Red=AfteR.
            updated_image_copy = (d.left_image).copy()
            cv2.circle(img=updated_image_copy, center=(cX,cY), radius=6, color=(255,0,0), thickness=-1)
            cv2.putText(img=updated_image_copy, text="{}".format((cX,cY)), org=(cX,cY), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)

            # Now we apply the callback and drag a box around the end-effector on the (updated!) image.
            window_name = "Robot has tried to move to ({},{}). Click and drag a box around its end effector. Then press any key.".format(cX,cY)
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, click_and_crop)
            cv2.imshow(window_name, updated_image_copy)
            key = cv2.waitKey(0)

            # Now save the image with the next available index. It will contain the contours.
            index = len(os.listdir(IMDIR))
            cv2.imwrite(IMDIR+"/point_"+str(index).zfill(2)+".png", updated_image_copy)
            cv2.destroyAllWindows()
 
            # Get position and orientation of the arm, save, & reset. Be careful that I understand `data_pt` ordering!
            frame = arm.get_current_cartesian_position()
            pos = tuple(frame.position[:3])
            rot = tfx.tb_angles(frame.rotation)
            rot = (rot.yaw_deg, rot.pitch_deg, rot.roll_deg)
            data_pt = (pos, rot, (cX,cY), CENTER_OF_BOXES[-1]) # Save the center of the bounding box.
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
