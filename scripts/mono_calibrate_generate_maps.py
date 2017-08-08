"""
Given pickle files from `mono_calibrate_onearm.py`, figure out the regressors for each camera. 

Edit: for now just do one arm.

Use this to also print and save the statistics of the data to GitHub for future reference.
"""

import environ
import pickle
import numpy as np
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
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

    Returns (X,Y) where X[i] is the i-th data point in terms of (x,y) pixel
    location in whatever camera the data was from, and Y[i] consists of the
    targets we want for both arms, with Y[i,0:2] for arm1 and Y[i,2:4] for arm2.
    Here, arm1, arm2 are left and right arms (but somewhat confusingly, they're 
    actually right and left from my perspective). 

    We only use two targets for each arm because (a) we ignore rotation, and (b)
    we ignore height, I think. It's easiest to assume fixed height anyway.

    EDIT: again never mind, one arm, sorry for the confusion ...
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

    print("for {}, avg(|| ytarg-ypred ||_2^2) = {:.7f}".format(arm, avg_l2))
    if do_validation:
        return reg, avg_l2
    else:
        return reg


def train_rigid_body(X, Y):
    """ Let's do something simpler instead of random forests. This is the rigid body
    movement problem, can be solved with Singular Value Decomposition.

    Update: Um ... this doesn't look too good. :-( Say, should we be normalizing data?
    """
    X = X.T # (2,N)
    Y = Y.T # (2,N)
    EX = np.mean(X, axis=1, keepdims=True) # (2,1)
    EY = np.mean(Y, axis=1, keepdims=True) # (2,1)
    A = X - EX # (2,N)
    B = Y - EY # (2,N)
    C = B.dot(A.T) # (2,2)
    U, s, VT = np.linalg.svd(C) # I think we can just use VT directly (VT = V^T)...
    R = U.dot(np.diag([1, np.linalg.det(U.dot(VT))])).dot(VT)
    d = EY - R.dot(EX)

    assert X.shape[0] < X.shape[1] and Y.shape[0] < Y.shape[1]
    assert A.shape == X.shape and B.shape == Y.shape
    assert R.shape == (2,2)
    assert d.shape == (2,1)
    Y_pred = R.dot(X) + d
    error_L2 = np.linalg.norm(Y_pred - Y)

    print("X:\n{}".format(X))
    print("R.dot(X):\n{}".format(R.dot(X)))
    print("R.dot(X)+d:\n{}".format(R.dot(X)+d))
    print("Y:\n{}".format(Y))
    print("R:\n{}".format(R))
    print("d:\n{}".format(d))
    print("(R.T).dot(R):\n{}".format((R.T).dot(R)))
    print("error_L2: {}".format(error_L2))


def train_least_squares(X, Y, X_mean=None, X_std=None):
    """ X,Y each have shape (N,2). """
    if X_mean is not None:
        X = (X - X_mean) / X_std
    lreg = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
    lreg.fit(X,Y)
    Y_pred = lreg.predict(X)
    error_L2 = np.linalg.norm(Y_pred - Y)
    print("\nLeast squares error_L2: {}".format(error_L2))
    return lreg


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
    num_trees = 100
    data = loadData('config/daniel_final_calib_v01.p')
    print_statistics(data)
    X,Y = dataToMatrix(data)
    print("X.shape: {}".format(X.shape))
    print("Y.shape: {}".format(Y.shape))
    pp = np.random.permutation(len(X))
    X = X[pp]
    Y = Y[pp]
    X_mean = np.mean(X, axis=0)
    X_std  = np.std(X, axis=0)
    assert X.shape[1] == 2 and Y.shape[1] == 2 # Both are (N,2).
    assert len(X_mean) == len(X_std) == 2 # Both are (2,).
    print("X_mean: {}".format(X_mean.T))
    print("X_std:  {}".format(X_std.T))

    # Random forests.
    #regl = train(X, Y, n_estimators=num_trees)
    #pickle.dump(regl, open('config/daniel_final_mono_map_01.p','wb'))

    # Various forms of least squares.
    #regressor_1 = train_rigid_body(X, Y)
    regressor_2 = train_least_squares(X, Y)
    regressor_3 = train_least_squares(X, Y, X_mean=X_mean, X_std=X_std)
    #pickle.dump(regressor_1, open('config/daniel_final_mono_rigidbody_01.p','wb'))
    pickle.dump(regressor_2, open('config/daniel_final_mono_lin-regression_01.p','wb'))
    pickle.dump(regressor_3, open('config/daniel_final_mono_lin-regression-normalizedX_01.p','wb'))
    pickle.dump((X_mean,X_std), open('config/mean_X,std_X.p','wb'))
