"""
Double check all my calibration points. Purely a debugging method. Also should be used 
to convert stuff into one list from the original pickle files.
"""

import cv2
import numpy as np
import pickle
import sys
import time
import utilities as utils

# 0X for grippers, 1X for the scissors
VERSION = '01'


def show_points(image, left):
    name = 'config/grid/calib_circlegrid_right_v'+VERSION+'_ONELIST.p' 
    if left:
        name = 'config/grid/calib_circlegrid_left_v'+VERSION+'_ONELIST.p' 
    contours = pickle.load(open(name, 'r'))

    for i,(pos,rot,cx,cy) in enumerate(contours):
        cv2.putText(img=image, text='{},{}'.format(cx,cy), org=(cx,cy),       
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(0,0,0), thickness=2)
        cv2.putText(img=image, text='{}'.format(i), org=(cx-5,cy-5),    
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(255,0,0), thickness=2)

    cv2.imshow("Image w/points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def turn_into_one_list():
    l_list = utils.pickle_to_list('config/grid/calib_circlegrid_left_v'+VERSION+'.p')
    r_list = utils.pickle_to_list('config/grid/calib_circlegrid_right_v'+VERSION+'.p')
    assert len(l_list) == len(r_list)
    assert 35 <= len(l_list) <= 36
    pickle.dump(l_list, open('config/grid/calib_circlegrid_left_v'+VERSION+'_ONELIST.p', 'w'))
    pickle.dump(r_list, open('config/grid/calib_circlegrid_right_v'+VERSION+'_ONELIST.p', 'w'))


if __name__ == "__main__":
    turn_into_one_list()
    arm1, _, d = utils.initializeRobots()
    show_points(d.left_image.copy(),  left=True)
    show_points(d.right_image.copy(), left=False)
