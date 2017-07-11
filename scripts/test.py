"""
Purely a testing script for me (Daniel Seita) to see how things work.

PS: be careful about PSM1 vs PSM2, easy to get confused. 
Don't forget also to turn on the endoscope using the other computer!
Otherwise the DataCollector's images will not exist. :)
"""

from autolab.data_collector import *
from dvrk.robot import *
from config.constants import *
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
print("\npsm1 current position: {}".format(psm1.get_current_cartesian_position()))
print("psm2 current position: {}".format(psm2.get_current_cartesian_position()))
psm1.home()
psm2.home()
print("\nJust moved to the home positions. Updated locations:")
print("psm1 current position: {}".format(psm1.get_current_cartesian_position()))
print("psm2 current position: {}".format(psm2.get_current_cartesian_position()))

print("d.left_image = {}".format(d.left_image))

"""
#move robot to a position
post,rott = ((0.05, 0.02, -0.15), (0.0, 0.0,-160.0))
pos = [post[0], post[1], post[2]]
rot = tfx.tb_angles(rott[0], rott[1], rott[2])
psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)
"""
