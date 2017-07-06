from dvrk.robot import *
from autolab.data_collector import *

import time


#initialize sensor readings
d = DataCollector()
print(d.identifier)

"""
time.sleep(2)


#initialize robot
psm1 = robot("PSM1")
psm1.home()

#move robot to a position
post,rott = ((0.05, 0.02, -0.15), (0.0, 0.0,-160.0))
pos = [post[0], post[1], post[2]]
rot = tfx.tb_angles(rott[0], rott[1], rott[2])
psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)
"""


