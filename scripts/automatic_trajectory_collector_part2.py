"""
Run this after running the automatic trajectory collector. This gets the data 
in the format I need for my old code. I need this separate for data cleaning, etc.

(c) September 2017 by Daniel Seita
"""

import cv2
import numpy as np
import os
import pickle
import sys
import utilities as utils
from autolab.data_collector import DataCollector
from dvrk.robot import *
np.set_printoptions(suppress=True)


def filter_points_in_results():
    """ Iterate through all trajectories to concatenate the data.

    Stuff to filter:

    (1) Both a left and a right camera must actually exist. In my code I set invalid
        ones to have (-1,-1) so that's one case.

    (2) Another case would be if the left-right transformation doesn't look good. Do
        this from the perspective of the _left_ camera since it's usually better. In
        other words, given pixels from the left camera, if we map them over to the
        right camera, the right camera's pixels should be roughly where we expect. If
        it's completely off, something's wrong.

    (3) ...
    """
    # Iterate through all directories, then we're good?
    pass


def process_data():
    """ Processes the filtered, cleaned data into something I can use in my old code.

    TODO
    """
    pass


if __name__ == "__main__":
    # Load the parameters from left to right camera that I did earlier.
    filter_points_in_results()
    # Now turn it into matrices I need
    process_data()
