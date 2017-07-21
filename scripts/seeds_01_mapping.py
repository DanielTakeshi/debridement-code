"""
Given pickle files from the open loop for seeds, figure out the regressor.
Not totally sure how this will work but I think it's OK if we just do mappings from
(x1,y1) -> (x2,y2), where these are points in the robot space, not the pixel space.
Look at `load_open_loop_data` to see how I stored the data.
"""

import environ
import pickle
import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor
np.set_printoptions(suppress=True, linewidth=200)

#####################
# Change stuff here #
#####################
num_trees = 5
data_file = 'data/demos_seeds_01.p'
out_file  = 'data/demos_seeds_01_mapping.p'
 

def load_open_loop_data(filename):
    """ Load data and manipulate it somehow and return the X, Y stuff. """
    data = []
    f = open(filename,'r')

    while True:
        try:
            d = pickle.load(f)
            assert len(d) == 4 # Because I stored four things (due to 4 seeds)
            for item in d:
                frame_before, frame_after, camera_pt = item
                frame_before = list(frame_before.position)
                frame_after  = list(frame_after.position)
                assert len(frame_before) == 3
                assert len(frame_after) == 3
                data.append( [frame_before[0], frame_before[1], frame_after[0], frame_after[1]] )
        except EOFError:
            break

    data = np.array(data)
    print("Here's our loaded data:\n{}".format(data))
    print("shape: {}".format(data.shape))
    X = data[:, :2]
    Y = data[:, 2:]
    return X, Y


def train(X_train, Y_train, n_estimators):
    """ Yeah, just fits X,Y. Simple. """
    reg = RandomForestRegressor(n_estimators=n_estimators)
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    avg_l2_train = np.sum((Y_pred-Y_train)*(Y_pred-Y_train), axis=1)
    avg_l2 = np.mean(avg_l2_train)
    print("avg(|| ytarg-ypred ||_2^2) = {:.6f}".format(avg_l2))
    return reg


if __name__ == "__main__":
    X_train, Y_train = load_open_loop_data(data_file)
    rf = train(X_train, Y_train, num_trees)
    pickle.dump(rf, open(out_file,'wb'))
