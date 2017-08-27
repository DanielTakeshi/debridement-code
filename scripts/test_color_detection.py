""" 
Test HSV saturation or other color detection stuff. I should at least explore this 
in case it allows us to do automatic color detection.
"""

import cv2
import image_geometry
import numpy as np
import os
import pickle
import sys
import utilities
from autolab.data_collector import DataCollector
from dvrk.robot import *
np.set_printoptions(suppress=True)


def detect_color_direct(image):
    """
    http://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/

    These are Adrian's boundaries in BGR format:

        boundaries = [
            ([17, 15, 100],  [50, 56, 200]),  # Red
            ([86, 31, 4],    [220, 88, 50]),  # Blue
            ([25, 146, 190], [62, 174, 250]), # Yellow
            ([103, 86, 65],  [145, 133, 128]) # Gray
        ]

    Unfortunately these are hard to get. These assume BGR format, btw ... but I tried
    switching the B and R values for the Yellow stuff and that doesn't work either. :-(
    When OpenCV reads in images from disk (e.g. `cv2.imread(...)`) it's assumed to be BGR,
    but I don't think that's true if I have `self.bridge.imgmsg_to_cv2(msg, "rgb8")`.
    """
    lower  = np.array([190, 146, 25], dtype='uint8')
    upper  = np.array([250, 174, 62], dtype='uint8')
    mask   = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)
    utilities.call_wait_key(cv2.imshow("left image", output))
    return output


def detect_color_hsv(frame):
    """ 
    The original image is stored in RGB (not BGR as is usual) because of the way
    we designed the autolab data collector. Hence, the use of `cv2.COLOR_RGB2HSV`.

    The color ranges are tricky. Use the stuff from:
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#object-tracking
    where they recommend having a single target for the hue value and then taking
    a range of +/- 10 about the center.

    My heuristics:

    Yellow tape:
        lower = np.array([0, 90, 90])
        upper = np.array([80, 255, 255])

    Red tape:
    """
    # Define the range of the color. TRICKY! Probably can't use these.
    colors = {
            'yellow': 30,
            'green':  60,
            'blue':   120
    }
    lower = np.array([110, 90, 90])
    upper = np.array([180, 255, 255])

    # Convert from RGB (not BGR) to hsv and apply our chosen thresholding.
    hsv  = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    utilities.call_wait_key(cv2.imshow("Does the mask make sense?", mask))
    res  = cv2.bitwise_and(frame, frame, mask=mask)

    # Let's inspect the output and pray it works.
    utilities.call_wait_key(cv2.imshow("Does it detect the desired color?", res))
    return res


def find_specific_spot(image):
    """ 
    Given thresholded image, have to figure out _where_ the end-effector tip is located.
    Parameter `image` should be a thresholded image from HSV stuff.
    To make it easier we should also have a bounding box condition.
    Note: to detect contours, we need grayscale (monochrome) images.
    """
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    img = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    utilities.call_wait_key(cv2.imshow("Grayscale image", image))

    # Detect contours *inside* the bounding box heuristic.
    xx, yy, ww, hh = 650, 50, 800, 800        
    (cnts, _) = cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contained_cnts = []
    print("number of cnts: {}".format(len(cnts)))

    for c in cnts:
        try:
            # Find the centroids of the contours in _pixel_space_. :)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Enforce it to be within bounding box.
            print(xx,cX,yy,cY)
            if (xx < cX < xx+ww) and (yy < cY < yy+hh):
                print("appending!")
                contained_cnts.append(c)
        except:
            pass
    print("number of contained contours: {}".format(len(contained_cnts)))

    contours = sorted(contained_cnts, key=cv2.contourArea, reverse=True)[:1]
    processed = []
    for c in contours:
        try:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            processed.append( (cX,cY,approx,peri) )
        except:
            pass
    print("number of processed contours: {}".format(len(processed)))

    for i, (cX, cY, approx, peri) in enumerate(processed):
        cv2.circle(img, (cX,cY), 50, (0,0,255))
        cv2.drawContours(img, [approx], -1, (0,255,0), 3)
        cv2.putText(img=img, text=str(i), org=(cX,cY), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=(0,0,0), thickness=2)
    utilities.call_wait_key(cv2.imshow("With contours!", img))
 

if __name__ == "__main__":
    arm, _, d = utilities.initializeRobots()
    arm.close_gripper()
    print("current arm position: {}".format(arm.get_current_cartesian_position()))
    utilities.call_wait_key(cv2.imshow("Bounding Box for Contours", d.left_image_bbox))
    
    frame = d.left_image.copy()

    res = detect_color_hsv(frame)
    #detect_color_direct(frame)
    find_specific_spot(res)
