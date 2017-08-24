""" 
Once we've manually run a bunch of stuff in `click_and_crop.py`, this takes that saved 
pickle file and plots things together into one plot. The data was stored like this:

data_pt = {'target_pos': pos,
           'target_rot': ROTATION,
           'actual_pos': new_pos,
           'actual_rot': new_rot,
           'center_target_pixels': (cX,cY),
           'center_actual_pixels': CENTER_OF_BOXES[-1]}

I.e. each element in the pickle file is a python dictionary.
"""

import cv2
import numpy as np
import os
import pickle
import sys

# Double check these as needed, especially the versions.
VERSION         = '11'
ORIGINAL_IMAGE  = 'images/left_image.jpg'
IM_VISUALS_DIR  = 'images/visuals/'
IM_VISUALS_FILE = 'config/calibration_results/data_v'+VERSION+'.p'


def load_data(filename):
    data = []
    f = open(filename,'rb')
    while True:
        try:
            d = pickle.load(f)
            assert len(d.keys()) == 6
            data.append( (d['center_target_pixels'], d['center_actual_pixels']) )
        except EOFError:
            break
    return data


def collect_stats(visuals):
    """ For distances, we use the usual distance formula for two points. """
    l2_distances = []
    for index,val in enumerate(visuals):
        targ, real = val
        dist = np.sqrt( (targ[0]-real[0])**2 + (targ[1]-real[1])**2 )
        l2_distances.append(dist)
    l2_distances = np.array(l2_distances)
    print("Distances among the pixels, (x,y) only:")
    print("mean:   {}".format(l2_distances.mean()))
    print("median: {}".format(np.median(l2_distances)))
    print("max:    {}".format(l2_distances.max()))
    print("min:    {}".format(l2_distances.min()))
    print("std:    {}".format(l2_distances.std()))


def make_fancy_image(image, visuals):
    """ Make a fancy image, e.g. for a paper or presentation.

    Parameters
    ----------
    image:
        A clone of the original image for calibration, can draw on it.
    visuals:
        Each item consists of *two* tuples, for target and actual, respectively.
    """
    radius = 7
    im_v01 = image.copy()

    for index,val in enumerate(visuals):
        pt_targ, pt_real = val
        cv2.circle(img=im_v01, center=pt_targ, radius=radius, color=(255,0,0), thickness=-1)
        cv2.circle(img=im_v01, center=pt_real, radius=radius, color=(0,0,255), thickness=-1)
        # Argh, this is not part of the interface, I think if I upgrade OpenCV I can do this.
        # cv2.arrowedLine(img=im_v01, pt1=pt_targ, pt2=pt_real, color=(0,0,0), thickness=1)
        cv2.line(img=im_v01, pt1=pt_targ, pt2=pt_real, color=(0,0,0), thickness=2)

    cv2.imshow("Image v01", im_v01)
    key = cv2.waitKey(0)
    cv2.imwrite(IM_VISUALS_DIR+'calib_image_v'+VERSION+'_yeslines.png', im_v01)


if __name__ == "__main__":
    image_original = cv2.imread(ORIGINAL_IMAGE).copy()
    visuals_data = load_data(IM_VISUALS_FILE)
    for item in visuals_data:
        print item
    print(len(visuals_data))
    collect_stats(visuals_data)
    make_fancy_image(image_original, visuals_data)
