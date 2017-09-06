"""
Given data from `scripts/human_guidance_auto.py`, we create regressors for this, typically
random forests since Deep Neural Networks are not needed with precious, human data.

Right now these mappings are for going from:

    (rx,ry,rz,yaw(?)) -> (change_x,change_y,change_x)

where rx,ry,rz are the *predicted* robot points, the ones where we think we originally want
to go to based on Deep Learning. This is NOT the same as the actual robot position encountered,
but we needed that so that we could get accurate targets. BTW, for these, we can assume a fixed
yaw so these are conditioned on a yaw, and if we don't assume a fixed yaw, we should take a
combination of the yaws so this will be like a meta-predictor. For now just think of the input
as going from 3D to 3D, _conditioned_ on a yaw.

Do not run this with command line arguments, because I think it will get confusing quickly.
Just check that the settings in the main method make sense.
"""

from dvrk.robot import *
from autolab.data_collector import DataCollector
from sklearn.ensemble import RandomForestRegressor
import cv2
import numpy as np
import os
import pickle
import sys
import time
import utilities as utils
np.set_printoptions(suppress=True)


def estimate_rf(X_train, Y_train, num_trees, debug=True):
    """ Hopefully this will narrow down errors even further.
    
    X_train: the robot coordinates, an array of (x,y,z) stuff, which was predicted
        by the Deep Neural Network. We don't use yaw, pitch, roll as input.
    Y_train: the residuals. When I got the rigid body transform points, I had to 
        correct for the points manually. So we have (xp,yp,zp) and (xt,yt,zt),
        and the residual is (xp-xt,yp-yt,zp-zt). Note, however, that I ignore the
        z-coordinate so the residual target should be an array of (eps_x, eps_y).
        EDIT: actually sometimes I ignore it, sometimes I don't!
            
    In real application, we need to get our camera points. Given those camera points,
    apply the neural net to get the predicted robot point (xp,yp,zp,yaw) with the
    pitch and roll interpolated from the yaw. Then with that yaw, we pick the closest
    RF for that (or we can interpolate, etc.). The RF trained here can predict 
    (xp-xt,yp-yt,zp-zt). Thus, we have to do:

        (xp,yp,zp) - (xp-xt,xp-xt,zp-zt)

    to get the new, more robust (hopefully) predictions. E.g. if xerr is 5mm, we 
    keep over-shooting x by 5mm so better bring it down. I've observed, for instance,
    that typicall our y-values are much too "high" (from the perspective of me 
    sitting by the platform). This means the values are negative as we subtract them,
    so the change is now positive and that makes sense as the y-axis is actually
    increasing *towards* me. The x-axis increases towards the wall, FYI.
    """
    assert X_train.shape[1] == 3 # (N,3)
    assert 2 <= Y_train.shape[1] <= 3 # (N,2) or (N,3)
    assert X_train.shape[0] == Y_train.shape[0]

    rf = RandomForestRegressor(n_estimators=num_trees)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_train)
    abs_errors = np.abs(Y_pred-Y_train)
    abs_mean_err = np.mean(abs_errors, axis=0)

    if debug:
        print("Begin debug prints for RFs:")
        print("X_train:\n{}".format(X_train))
        print("Y_train:\n{}".format(Y_train))
        print("Y_pred:\n{}".format(Y_pred))
        print("abs_mean_err: {}".format(abs_mean_err))
        print("End debug prints for RFs\n")
    return rf


if __name__ == "__main__":
    """ Double check all these arguments! Let's just do them in bulk mode. 
    
    For now I have six indiviual RFs. To make a meta RF with the 5 RFs corresponding
    to yaw values, first detect for a yaw. Then choose which of the five yaw RFs to use. 
    """

    # UPDATE: this is the "second" version, i.e. v01 which I did September 6.
    VERSION = 1
    num_trees = 100         # Since this seems like a good number.
    if VERSION == 0:
        nums = [str(x) for x in range(20,25+1)]
        yaws = [90, 45, 0, -45, -90, None]
    elif VERSION == 1:
        nums = [str(x) for x in range(40,44+1)]
        yaws = [90, 45, 0, -45, -90]
    
    names_input     = ['config/calibration_results/data_human_guided_v'+vv+'.p' for vv in nums]
    names_output_v0 = ['config/mapping_results/rf_human_guided_yesz_v'+vv+'.p' for vv in nums]
    names_output_v1 = ['config/mapping_results/rf_human_guided_noz_v'+vv+'.p' for vv in nums]
    
    for ii,(name,yaw) in enumerate(zip(names_input,yaws)):
        print("\nOn yaw: {}".format(yaw))
        data = utils.pickle_to_list(name)
        assert len(data) == 35
        outfile_v0 = open(names_output_v0[ii], 'w')
        outfile_v1 = open(names_output_v1[ii], 'w')

        # The PREDICTED (and not the measured values after moving) is the RF input.
        X_train = [data_pt['original_robot_point_prediction'] for data_pt in data]
        before  = [data_pt['measured_robot_point_before_human_change'].position[:3] for data_pt in data]
        after   = [data_pt['measured_robot_point_after_human_change'].position[:3]  for data_pt in data]

        X_train = np.squeeze(np.array(X_train))
        before  = np.squeeze(np.array(before))
        after   = np.squeeze(np.array(after))

        # Here, I think the z-coordinate is more important so let's test with and without.
        errors_v0 = (before - after)        # Keep z 
        errors_v1 = (before - after)[:, :2] # Ignore z

        # Build RFs and dump to the appropriate files. Later, I can build a meta-RF
        # by knowing which values map to the correct yaw values.
        rf_v0 = estimate_rf(X_train=X_train, Y_train=errors_v0, num_trees=num_trees)
        rf_v1 = estimate_rf(X_train=X_train, Y_train=errors_v1, num_trees=num_trees)
        pickle.dump(rf_v0, outfile_v0)
        pickle.dump(rf_v1, outfile_v1)
        outfile_v0.close()
        outfile_v1.close()
