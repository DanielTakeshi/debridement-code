"""
Given pickle files from `mono_calibrate.py`, figure out the regressors for each camera.
Edit: for now just do one arm.
"""

import environ
import pickle
import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor
np.set_printoptions(suppress=True, linewidth=200)


def loadData(filename):
    data = []
    f = open(filename,'r')
    while True:
        try:
            d = pickle.load(f)
            if not outlier(d):
                data.append(d)
        except EOFError:
            break
    return data


def outlier(d):
    """ I think this is roughly the height that we should expect. These values are
    similar to what's in the `constants.py` file.
    """
    pos1, _ , cX, cY = d
    if np.ravel(pos1[2])[0] > -0.135:
        return True
    else:
        return False


def dataToMatrix(data):
    """ Here, `data` is a pickle file corresponding to data from _one_ camera,
    either left or right, but it has information from both _arms_.

    EDIT: nope, never mind, it's one arm now. Sorry for the confusion.
    
    Returns (X,Y) where X[i] is the i-th data point in terms of (x,y) pixel
    location in whatever camera the data was from, and Y[i] consists of the
    targets we want for both arms, with Y[i,0:2] for arm1 and Y[i,2:4] for arm2.
    Here, arm1, arm2 are left and right arms (but somewhat confusingly, they're 
    actually right and left from my perspective). 

    EDIT: again never mind, one arm, sorry for the confusion ...
    
    We only use two targets for each arm because (a) we ignore rotation, and (b)
    we ignore height, I think. It's easiest to assume fixed height anyway.
    """
    X = np.zeros((len(data),2))
    Y = np.zeros((len(data),2))
    
    for i, arm1 in enumerate(data):
        pos1, _ , cX, cY = arm1
        Y[i,0] = np.ravel(pos1[0])[0]
        Y[i,1] = np.ravel(pos1[1])[0]
        X[i,0] = cX
        X[i,1] = cY
    return X,Y


def train(X, Y, arm=' left', X_valid=None, Y_valid=None, n_estimators=10):
    """ 
    Handles both the case when I have one training set and when I do k-fold cross validation.
    Also returns the avg_l2 error on the validation set.
    """
    do_validation = True if X_valid is not None else False
    reg = RandomForestRegressor(n_estimators=n_estimators)
    reg.fit(X,Y)

    if do_validation:
        Y_pred = reg.predict(X_valid)
        avg_l2_train = np.sum((Y_pred-Y_valid)*(Y_pred-Y_valid), axis=1)
    else:
        Y_pred = reg.predict(X)
        avg_l2_train = np.sum((Y_pred-Y)*(Y_pred-Y), axis=1)
    avg_l2 = np.mean(avg_l2_train)

    print("for {}, avg(|| ytarg-ypred ||_2^2) = {:.6f}".format(arm, avg_l2))
    if do_validation:
        return reg, avg_l2
    else:
        return reg


def print_statistics(data):
    """ Debugging method to help me figure out what height to use, for instance.
    Note again that we're assuming fixed heights. 
    """
    pos1_list = []

    for i, arm1 in enumerate(data):
        pos1, _ , cX, cY = arm1
        pos1_list.append(list(pos1))

    all_pos1 = np.array(pos1_list)
    print("all_pos1.shape: {}".format(all_pos1.shape))
    print("  (all_pos1)")
    print("mean {}  std {}".format(np.mean(all_pos1, axis=0).T, np.std(all_pos1, axis=0).T))
    print("max  {}  min {}".format(np.max(all_pos1, axis=0).T,  np.min(all_pos1, axis=0).T))


if __name__ == "__main__":
    """ Do regression on one arm. """
    num_trees = 5
    data = loadData('config/daniel_final_calib_v00.p')
    print_statistics(data)
    X,Y = dataToMatrix(data)
    pp = np.random.permutation(len(X))
    X = X[pp]
    Y = Y[pp]
    regl = train(X, Y[:,0:2], n_estimators=num_trees)
    pickle.dump(regl, open('config/daniel_final_mono_map_00.p','wb'))
