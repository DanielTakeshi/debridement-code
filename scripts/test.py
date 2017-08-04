"""
Purely a testing script for me (Daniel Seita) to see how things work. Be careful about PSM1 vs PSM2, 
easy to get confused.  Don't forget also to turn on the  endoscope using the other computer! Otherwise 
the DataCollector's images will not exist. :)

Also, note that yaw, pitch, and roll are modified when you spin the PSM arms. Otherwise, for 
actual motion planning, it seems like it's OK to assume that we use the "home" (yaw,pitch,roll) 
setting for each arm. We insert that, along with the (x,y,z) stuff, into a tfx data structure 
which gives us what we need.

Questions/observations:

    (1) Why is psm2.get_current_joint_position() = []?
    (2) When I make the psm1 have a roll of 0.0, why does it fail? For smaller roll changes, it seems OK.
    (3) The "roll" to me actually looks like "yaw" for some reason if we view the long arm as an airplane.
    (4) Fortunately, "pitch" has the intuitive interpretation, and "yaw" looks like the "roll." Yeah...

In terms of angle limits, looks like (at least on the left arm, psm1):

    yaw:   [-180, 180] # I think this has the full range of 360 degrees of motion. Good!
    pitch: [-50, 50] # Limited, intuitively like moving wrist up and down. due to mov
    roll:  [-180, -100] # Limited, intuitively like moving a wrist sideways, as in "yaw" I know...
"""

from autolab.data_collector import *
from dvrk.robot import *
from config.constants import *
import sys
import time
import cv2
import numpy as np
ESC_KEY = 27


def initializeRobots():
    d = DataCollector()
    r1 = robot("PSM1") # left (but my right)
    r2 = robot("PSM2") # right (but my left)
    time.sleep(2)
    return (r1,r2,d)


def call_wait_key(nothing=None):
    """ I have an ESC which helps me exit program. """
    key = cv2.waitKey(0)
    if key == ESC_KEY:
        print("Pressed ESC key. Terminating program...")
        sys.exit()


right_clicks = []
def mouse_callback(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONCLICK:
        #store the coordinates of the right-click event
        right_clicks.append([x, y])
        #this just verifies that the mouse data is being collected 
        #you probably want to remove this later 
        print right_clicks
    else:
        print("event: {}".format(event))


#########################
## TESTING BEGINS HERE ##
#########################

arm1, arm2, d = initializeRobots()
arm1.home()
arm2.home()
#call_wait_key(cv2.imshow("Left Image",  d.left_image))
image = d.left_image.copy()

cv2.namedWindow('image')
cv2.setMouseCallback('image', mouse_callback)
cv2.imshow("Image result", image)
cv2.waitKey(0)

print("right_clicks: {}".format(right_clicks))
sys.exit()






# Quick test
psm1.close_gripper()
rot = (0.0, 0.0, -160.0)
rot = tfx.tb_angles(rot[0], rot[1], rot[2])

pos1 = [0.020, 0.081, -0.16]
psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos1, rot), 0.03)
psm1.home()
psm1.close_gripper()
psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos1, rot), 0.03)
psm1.home(open_gripper=False)
sys.exit()

print("\npsm1 current position: {}".format(psm1.get_current_cartesian_position()))
print("psm2 current position: {}".format(psm2.get_current_cartesian_position()))
print("psm1 current JOINT position: {}".format(psm1.get_current_joint_position()))
print("psm2 current JOINT position: {}".format(psm2.get_current_joint_position()))
psm1.home()
psm2.home()
print("\nJust moved to the home positions. Updated locations:")
print("psm1 current position: {}".format(psm1.get_current_cartesian_position()))
print("psm2 current position: {}\n".format(psm2.get_current_cartesian_position()))

# Test rotations for PSM1. Note that: HOME_POSITION_PSM1 = ((0.00, 0.06, -0.13), (0.0, 0.0,-160.0))
def move(i,p,r):
    pos = [p[0], p[1], p[2]]
    rot = tfx.tb_angles(r[0], r[1], r[2])
    psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), SAFE_SPEED)
    time.sleep(2)
    print("Position {}: {}".format(str(i).zfill(2), psm1.get_current_cartesian_position()))

positions = [
    ((0.00, 0.06, -0.13), (0.0, 0.0, -160.0)),
    ((0.00, 0.06, -0.13), (0.0, 50.0, -160.0)),
    ((0.00, 0.06, -0.13), (0.0, -50.0, -160.0)),
    ((0.00, 0.06, -0.13), (0.0, 0.0, -160.0)) # Home it back
]

for i,(p,r) in enumerate(positions):
    move(i,p,r)
