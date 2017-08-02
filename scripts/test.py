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


# Initialize sensor readings
d = DataCollector()
print("Unique identifier, could be useful: {}".format(d.identifier))


# Initialize robot and move to home positions. Be careful, easy to forget if using "1" or "2". (:
time.sleep(3)
psm1 = robot("PSM1")
psm2 = robot("PSM2")
print("\nOfficial home position for psm1: {}".format(HOME_POSITION_PSM1))
print("Official home position for psm2: {}".format(HOME_POSITION_PSM2))


# Quick test
rot = (0.0, 0.0, -160.0)
rot = tfx.tb_angles(rot[0], rot[1], rot[2])

pos1 = [0.020, 0.081, -0.16]
psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos1, rot), 0.03)
print(psm1.get_current_cartesian_position())

pos2 = [0.064, 0.041, -0.16]
psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos2, rot), 0.03)
print(psm1.get_current_cartesian_position())
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
