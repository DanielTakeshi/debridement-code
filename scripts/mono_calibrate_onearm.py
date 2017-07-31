"""
Note: this is for ONE arm, i.e. the left arm, PSM1.

This script handles the wrist calibration of the robot to image frame.
More precisely, I (Daniel Seita) will use this to figure out how to do 
camera calibration assuming a fixed height. I also assume that we use
an image consisting of a bunch of black dots, all at the same height.
These assumptions are not entirely realistic.

Note: camera calibration should be done with a clear image with lots of 
obvious circles, as this relies on detecting circles (at a fixed height).
Unfortunately, even with these circles, sometimes the contour detection
code is too aggressive and picks other circles that aren't desirable. In
those cases, USE THE ESCAPE KEY. Then it will not store that contour.

Advice:

    - Keep the surgical camera fixed! (It's adjustable, but stable.)
    - Do roughly one or two matchings per circle. If a contour repeats
            in the same spot, just press ESCAPE to save time.
    - Try not to move the gauze. If it's shifted, move it back to a 
            starting reference point.
    - The camera is unfortunately closer to the left arm, meaning that
            the right arm sometimes can't reach the far end.
    - Push the pickle files to GitHub as soon as possible!
"""

import environ
from dvrk.robot import *
from autolab.data_collector import DataCollector
import cv2
import numpy as np
import pickle
import sys
IMDIR = "scripts/images/"


def initializeRobots():
    d = DataCollector()
    r1 = robot("PSM1") # left (but my right)
    r2 = robot("PSM2") # right (but my left)
    time.sleep(2)
    return (r1,r2,d)


def storeData(filename, arm1):
    """ 
    Stores data by repeatedly appending arm1 data points for  the file. 
    Then other code can simply enumerate over the whole thing. This is
    with respect to a single camera, remember that.
    """
    f = open(filename, 'a')
    pickle.dump(arm1, f)
    f.close()


def calibrateImage(contours, img, arm1, filename):
    """ Perform camera calibration using a fixed height and one image.
    
    Important note! When calling `cv2.waitKey(0)`, that will make the 
    program stop until any key has been pressed. That's when a given 
    contour has been chosen and when we have to move the dvrk arm to 
    the appropriate spot, for camera calibraton. If the contour is not
    actually a contour, press the escape key to ignore this step.

    This code saves the camera pixels (cX,cY) and the robot coordinates 
    (the (pos,rot) for ONE arm) all in one pickle file. Then, not in
    this code, we run regression to get our desired mapping from pixel 
    space to robot space. Whew. It's manual, but worth it.
    """
    arm1.home()
    arm1.close_gripper()
    print("(after calling `home`) psm1 current position: {}".format(
        arm1.get_current_cartesian_position()))
    print("len(contours): {}".format(len(contours)))

    for i, (cX, cY, approx, peri) in enumerate(contours):  
        limg = img.copy()
        rimg = img.copy()

        # Deal with the left camera
        cv2.circle(limg, (cX,cY), 50, (0,0,255))
        cv2.drawContours(limg , [approx], -1, (0,255,0), 3)
        cv2.imshow("Left Calibration PSM", limg)
        key1 = cv2.waitKey(0) 

        if key1 != 27:
            # Get position and orientation of the arm, save, & reset.
            frame = arm1.get_current_cartesian_position()
            pos = tuple(frame.position[:3])
            rot = tfx.tb_angles(frame.rotation)
            rot = (rot.yaw_deg, rot.pitch_deg, rot.roll_deg)
            a1 = (pos, rot, cX, cY)
            print("contour {}, a1={}".format(i,a1))
        else:
            print("(not storing contour {} on the left)".format(i))
        arm1.home()
        arm1.close_gripper()

        # Only store this contour if both keys were not escape keys.
        if key1 != 27:
            storeData(filename, a1)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    arm1, _, d = initializeRobots()
    arm1.close_gripper()
    cv2.imwrite(IMDIR+"left_camera_image.png",  d.left_image)
    #cv2.imshow("Left Camera Image", d.left_image)
    #cv2.waitKey(0)

    # NOTE! IMPORTANT! CHANGE THIS!
    vv = str(0).zfill(2)

    # left and then right calibration
    calibrateImage(d.left_contours,  d.left_image,  arm1, 'config/daniel_final_calib_v'+vv+'.p')
