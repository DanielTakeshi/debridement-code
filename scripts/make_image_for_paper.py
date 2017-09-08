"""
This is for making images for the paper, i.e. pictures of the dVRK platform.
"""

from autolab.data_collector import DataCollector
from dvrk.robot import *
import cv2
import image_geometry
import numpy as np
import os
import pickle
import sys
import tfx
import utilities as utils
np.set_printoptions(suppress=True)


if __name__ == "__main__":
    """ See the top of the file for program-wide arguments. """
    arm, _, d = utils.initializeRobots(sleep_time=2)
    arm.open_gripper(90)
    #print(arm.get_current_cartesian_position())
    #utils.home(arm)
    #arm.close_gripper()

    # Image, showing the human can push and modify things. Just put it at a specific
    # angle and let the human move it.

    # Next image, let's get five rotations and yaws visualized. Just do one by one
    # since I need pictures ... Remember, it's -90, -45, 0, 45, and 90 that I used.
    yaw = 90
    pitch, roll = utils.get_interpolated_pitch_and_roll(yaw)
    print("yaw: {}\npitch: {}\nroll: {}".format(yaw,pitch,roll))
    current_pos, current_rot = utils.lists_of_pos_rot_from_frame(
            arm.get_current_cartesian_position()
    )
    arm.open_gripper(90)
    utils.move(arm, current_pos, [yaw, pitch, roll], 'Fast')

    new_pos, new_rot = utils.lists_of_pos_rot_from_frame(
            arm.get_current_cartesian_position()
    )
    print("New rotation: {}".format(new_rot))
