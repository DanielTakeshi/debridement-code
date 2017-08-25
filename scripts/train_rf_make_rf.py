"""
Given data from `scripts/train_rf.py`, we create a random forest for this.
For now we're going to do it (x,y,z) -> (x',y') and assume that the z coordinate is fine.
(The z coordinate was already fine when I tried out the rigid body transformation.)
"""

from dvrk.robot import *
from autolab.data_collector import DataCollector
from sklearn.ensemble import RandomForestRegressor
import cv2
import numpy as np
import pickle
import sys
import time
np.set_printoptions(suppress=True)

# DOUBLE CHECK ALL THESE!! Especially any version numbers.
VERSION = '10'
FILE_FOR_RF = open('config/calibration_results/data_for_rf_v'+VERSION+'.p', 'r')
OUTFILE_RF  = open('config/mapping_results/random_forest_predictor_v'+VERSION+'.p', 'w')


def estimate_rf(X_train, Y_train, debug=True):
    """ Hopefully this will narrow down errors even further.
    
    X_train: the robot coordinates, an array of (x,y,z) stuff, which represents the
        rigid body transform.
    Y_train: the residuals. When I got the rigid body transform points, I had to 
        correct for the points manually. So we have (xp,yp,zp) and (xt,yt,zt),
        and the residual is (xp-xt,yp-yt,zp-zt). Note, however, that I ignore the
        z-coordinate so the residual target should be an array of (eps_x, eps_y).
            
    In real application, we need to get our camera points. Given those camera points,
    apply the rigid body to get the predicted robot point (xp,yp,zp), assuming fixed
    rotation for now. But, this has some error from that in the x and y directions.
    The RF trained here can predict (xp-xt,yp-yt). Thus, we have to do:

        (xp,yp,zp) - (xp-xt,xp-xt,0)

    to get the new, more robust (hopefully) predictions. E.g. if xerr is 5mm, we 
    keep over-shooting x by 5mm so better bring it down.
    """
    assert X_train.shape[1] == 3 # (N,3)
    assert Y_train.shape[1] == 2 # (N,2)
    assert X_train.shape[0] == Y_train.shape[0]

    rf = RandomForestRegressor(n_estimators=100) # 100 seems like a good number.
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_train)
    abs_errors = np.abs(Y_pred-Y_train)
    abs_mean_err = np.mean(abs_errors, axis=0)

    if debug:
        print("\nBegin debug prints for RFs:")
        print("X_train:\n{}".format(X_train))
        print("Y_train:\n{}".format(Y_train))
        print("Y_pred:\n{}".format(Y_pred))
        print("abs_mean_err: {}".format(abs_mean_err))
        print("End debug prints for RFs\n")
    return rf


if __name__ == "__main__":
    data = []
    while True:
        try:
            data.append(pickle.load(FILE_FOR_RF))
        except EOFError:
            break
    assert len(data) == 36 or len(data) == 35 # For now

    # Remember, the _predicted_ (not the actual) is the _input_ to the RF.
    X_train = [data_pt['original_robot_point_prediction'] for data_pt in data]
    before  = [data_pt['measured_robot_point_before_human_change'].position[:3] for data_pt in data]
    after   = [data_pt['measured_robot_point_after_human_change'].position[:3]  for data_pt in data]

    X_train = np.squeeze(np.array(X_train))
    before  = np.squeeze(np.array(before))
    after   = np.squeeze(np.array(after))

    errors  = (before - after)[:, :2] # Ignore z coordinate
    rf = estimate_rf(X_train=X_train, Y_train=errors)
    pickle.dump(rf, OUTFILE_RF)
