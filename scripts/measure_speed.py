"""
Measure speed to report in the paper, because 0.03 and 0.06 m/s is really unrealiable.
This will involve me holding a stopwatch. Yeah.
"""

from autolab.data_collector import DataCollector
from dvrk.robot import *
import numpy as np
import pickle
import sys
import time
import tfx
import utilities

###########################
# ADJUST THESE AS NEEDED! #
###########################

VERSION_INPUT = '00'
RF_REGRESSOR  = pickle.load(open('config/mapping_results/random_forest_predictor_v'+VERSION_INPUT+'.p', 'r'))
PARAMETERS    = pickle.load(open('config/mapping_results/params_matrices_v'+VERSION_INPUT+'.p', 'r'))
ROTATION      = utilities.get_average_rotation(VERSION_INPUT)
TFX_ROTATION  = tfx.tb_angles(ROTATION[0], ROTATION[1], ROTATION[2])
HOME          = [0.00, 0.06, -0.13]
NUM_RUNS      = 10

##########################
# END OF `CONFIGURATION` #
##########################

def move(arm, pos, SPEED_CLASS):
    """ Handles the different speeds we're using. """
    if SPEED_CLASS == 'Slow':
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, TFX_ROTATION), 0.03)
    elif SPEED_CLASS == 'Medium':
        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, TFX_ROTATION), 0.06)
    elif SPEED_CLASS == 'Fast':
        arm.move_cartesian_frame(tfx.pose(pos, TFX_ROTATION))
    else:
        raise ValueError()


def benchmark_time(arm, START, TARGET, SPEED_CLASS):
    """ This method returns an estimate of the speed. """
    times = []
    for i in range(NUM_RUNS):
        move(arm, START, 'Slow') # Just bring it slowly to the starting position.
        time.sleep(2)
        start_time = time.time()
        move(arm, TARGET, SPEED_CLASS) # This is what we measure.
        difference = time.time() - start_time
        times.append(difference)
        time.sleep(2)
    times = np.array(times)
    print("\ntimes (in seconds!!) for speed class {}:\n{}".format(SPEED_CLASS, times))
    print("mean(times): {:.2f}".format(np.mean(times)))
    print("std(times):  {:.2f}".format(np.std(times)))


if __name__ == "__main__":
    """ See the top of the file for program-wide arguments. """
    arm, _, d = utilities.initializeRobots(sleep_time=2)
    arm.close_gripper()
    START = [HOME[0]-0.03, HOME[1], HOME[2]]
    TARGET = [START[0]+0.1, START[1], START[2]]
    benchmark_time(arm, START, TARGET, SPEED_CLASS='Slow')
    benchmark_time(arm, START, TARGET, SPEED_CLASS='Medium')
    benchmark_time(arm, START, TARGET, SPEED_CLASS='Fast')
