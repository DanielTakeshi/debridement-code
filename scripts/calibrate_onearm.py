"""
Note: this is for ONE arm, i.e. the left arm, PSM1, but with BOTH cameras!
I used to do this with a fixed height (z-coordinate) but that does not work.
One thing that's different here is that we need left/right contours to match.
Actually maybe that's not so bad? All we need is the pixel points and if we arrange
then from left to right, up to down then we know which ones corresponded to which.
Then the only way this usually fails is if I skipped a contour by mistake. 
Which I better not do!!

Advice:

    - Keep the surgical camera fixed! (It's adjustable, but stable.)
            AGAIN DO NOT CHANGE THE LOCATION!!
    - Do ONE calibration per circle, to avoid confusion with different camera images.
    - Try not to move the gauze. If it's shifted, move it back to a 
            starting reference point.
    - The camera is unfortunately closer to the left arm, meaning that
            the right arm sometimes can't reach the far end.
    - Push the pickle files to GitHub as soon as possible!

Note: when calling `cv2.waitKey(0)`, that will make the program stop until 
any key has been pressed. That's when a given contour has been chosen and 
when we have to move the dvrk arm to the appropriate spot, for camera calibraton. 
If the contour is not actually a contour, press the escape key to ignore this step.
"""

from dvrk.robot import *
from autolab.data_collector import DataCollector
import cv2
import numpy as np
import pickle
import sys

# IMPORTANT! CHANGE THESE!
VERSION  = str(0).zfill(2)
IMDIR    = "images/"
ESC_KEYS = [27, 1048603]


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


def calibrateImage(contours, img, arm1, outfile):
    """ Perform camera calibration using both images.
    
    This code saves the camera pixels (cX,cY) and the robot coordinates 
    (the (pos,rot) for ONE arm) all in one pickle file. Then, not in
    this code, we run regression to get our desired mapping from pixel 
    space to robot space. Whew. It's manual, but worth it. I put numbers
    to indicate how many we've saved. DO ONE SAVE PER CONTOUR so that I 
    can get a correspondence with left and right images after arranging
    pixels in the correct ordering (though I don't think I have to do that.).
    """
    arm1.home()
    arm1.close_gripper()
    print("(after calling `home`) psm1 current position: {}".format(
        arm1.get_current_cartesian_position()))
    print("len(contours): {}".format(len(contours)))
    num_saved = 0

    for i, (cX, cY, approx, peri) in enumerate(contours):  
        image = img.copy()

        # Deal with the image and get a visual.
        cv2.circle(image, (cX,cY), 50, (0,0,255))
        cv2.drawContours(image, [approx], -1, (0,255,0), 3)
        cv2.putText(img=image, text=str(num_saved), org=(cX,cY), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0,0,0), thickness=2)
        cv2.imshow("Contour w/{} saved so far out of {}.".format(num_saved,i), image)
        key1 = cv2.waitKey(0) 

        if key1 not in ESC_KEYS:
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
        if key1 not in ESC_KEYS:
            storeData(outfile, a1)
            num_saved += 1
        cv2.destroyAllWindows()


if __name__ == "__main__":
    arm1, _, d = initializeRobots()
    arm1.close_gripper()

    # Keep this in case I want to debug some images.
    #cv2.imwrite(IMDIR+"left_camera_image.png",  d.left_image)
    #cv2.imshow("Left Camera Image", d.left_image)
    #cv2.waitKey(0)

    # Calibrate using the left and then the right images.
    calibrateImage(contours = d.left_contours, 
                   img = d.left_image, 
                   arm1 = arm1, 
                   outfile = 'config/calib_circlegrid_left_v'+VERSION+'.p')
    calibrateImage(contours = d.right_contours, 
                   img = d.right_image, 
                   arm1 = arm1, 
                   outfile = 'config/calib_circlegrid_right_v'+VERSION+'.p')
