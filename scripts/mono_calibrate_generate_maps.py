"""
Given pickle files from `mono_calibrate.py`, figure out the regressors for each camera.
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
    arm1, arm2 = d
    pos1, _ , cX, cY = arm1
    pos2, _ , cX, cY = arm2
    if np.ravel(pos1[2])[0] > -0.135 or \
        np.ravel(pos2[2])[0] > -0.105:
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
    """
    X = np.zeros((len(data),2))
    Y = np.zeros((len(data),4))
    
    for i, d in enumerate(data):
        arm1, arm2 = d
        pos1, _ , cX, cY = arm1
        pos2, _ , cX, cY = arm2
        Y[i,0] = np.ravel(pos1[0])[0]
        Y[i,1] = np.ravel(pos1[1])[0]
        Y[i,2] = np.ravel(pos2[0])[0]
        Y[i,3] = np.ravel(pos2[1])[0]
        X[i,0] = cX
        X[i,1] = cY
    return X,Y


def train(X,Y):
    reg = RandomForestRegressor(n_estimators=10)
    reg.fit(X,Y)
    Y_pred = reg.predict(X)
    avg_l2_train = np.sum((Y_pred-Y)*(Y_pred-Y), axis=1)
    avg_l2_train = np.mean(avg_l2_train)
    print("For training, average(|| ytarg-ypred ||_2^2) = {}".format(avg_l2_train))
    return reg


if __name__ == "__main__":
    """ Do regression for both the left and right cameras, and for each, handle the
    two DVRK arms separately. """

    print("Now loading data from the left camera ...")
    data = loadData('config/daniel_left_camera.p')
    X,Y = dataToMatrix(data)
    regl = train(X,Y[:,0:2])
    regr = train(X,Y[:,2:4])
    pickle.dump((regl,regr), open('config/daniel_left_mono_model.p','wb'))

    print("Now loading data from the right camera ...")
    data = loadData('config/daniel_right_camera.p')
    X,Y = dataToMatrix(data)
    regl = train(X,Y[:,0:2])
    regr = train(X,Y[:,2:4])
    pickle.dump((regl,regr), open('config/daniel_right_mono_model.p','wb'))
