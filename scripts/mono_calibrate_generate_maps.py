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
    """ I think this is roughly the height that we should expect. These values are
    similar to what's in the `constants.py` file.
    """
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


def train(X, Y, arm, X_valid=None, Y_valid=None, n_estimators=10):
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
    pos2_list = []

    for i, d in enumerate(data):
        arm1, arm2 = d
        pos1, _ , cX, cY = arm1
        pos2, _ , cX, cY = arm2
        pos1_list.append(list(pos1))
        pos2_list.append(list(pos2))

    all_pos1 = np.array(pos1_list)
    all_pos2 = np.array(pos2_list)
    print("all_pos1.shape: {}".format(all_pos1.shape))
    print("all_pos2.shape: {}".format(all_pos2.shape))
    print("  (all_pos1)")
    print("mean {}  std {}".format(np.mean(all_pos1, axis=0).T, np.std(all_pos1, axis=0).T))
    print("max  {}  min {}".format(np.max(all_pos1, axis=0).T,  np.min(all_pos1, axis=0).T))
    print("  (all_pos2)")
    print("mean {}  std {}".format(np.mean(all_pos2, axis=0).T, np.std(all_pos2, axis=0).T))
    print("max  {}  min {}".format(np.max(all_pos2, axis=0).T,  np.min(all_pos2, axis=0).T))


if __name__ == "__main__":
    """ 
    Do regression for both the left and right cameras, and for each, handle the
    two DVRK arms separately. Also, do k-fold cross validation. 
    """
    # Some tunable hyper-parameters.
    kfolds = 5
    num_trees = 5
    print("Hyperparameters: kfolds {}, num_trees {}".format(kfolds, num_trees))

    ########
    # LEFT #
    ########
    print("\n\t\tNow loading data from the LEFT camera ...\n")
    data = loadData('config/daniel_left_camera_v02.p')+loadData('config/daniel_left_camera_v03.p')
    print_statistics(data)
    X,Y = dataToMatrix(data)
    pp = np.random.permutation(len(X))
    X = X[pp]
    Y = Y[pp]
    regl = train(X, Y[:,0:2], arm='left ', n_estimators=num_trees)
    regr = train(X, Y[:,2:4], arm='right', n_estimators=num_trees)
    print("(The above was for the full dataset ... now let's do k-folds to see generalization.)")
    pickle.dump((regl,regr), open('config/daniel_left_mono_model_v02_and_v03.p','wb'))

    num_valid = int(len(X) / kfolds)
    num_train = len(X) - num_valid
    best_loss = np.float('inf')
    best_idx = -1
    avg_loss = 0.0

    for k in range(kfolds):
        vstart = k*num_valid
        vend = (k+1)*num_valid
        X_valid = X[vstart:vend]
        Y_valid = Y[vstart:vend]
        X_train = np.concatenate((X[:vstart], X[vend:]))
        Y_train = np.concatenate((Y[:vstart], Y[vend:]))
        print("\n  On kfold {}, valid range {} to {} (inclusive,exclusive)".format(k, vstart, vend))
        print("with shapes X_train, X_valid = {}, {}".format(X_train.shape, X_valid.shape))
        regl, lossl = train(X_train, Y_train[:,0:2], arm='left ', X_valid=X_valid, Y_valid=Y_valid[:,0:2], n_estimators=num_trees)
        regr, lossr = train(X_train, Y_train[:,2:4], arm='right', X_valid=X_valid, Y_valid=Y_valid[:,2:4], n_estimators=num_trees)
        loss = (lossl + lossr) / 2. # Will average losses among arms
        avg_loss += loss
        if loss < best_loss:
            best_loss = loss
            best_idx = k
    avg_loss /= kfolds
    print("\nbest_loss: {:.6f} at index {}, w/avg_loss: {:.6f}".format(best_loss, best_idx, avg_loss))

    #########
    # RIGHT #
    #########
    print("\n\t\tNow loading data from the RIGHT camera ...\n")
    data = loadData('config/daniel_right_camera_v02.p')+loadData('config/daniel_right_camera_v03.p')
    print_statistics(data)
    X,Y = dataToMatrix(data)
    pp = np.random.permutation(len(X))
    X = X[pp]
    Y = Y[pp]
    regl = train(X, Y[:,0:2], arm='left ', n_estimators=num_trees)
    regr = train(X, Y[:,2:4], arm='right', n_estimators=num_trees)
    print("(The above was for the full dataset ... now let's do k-folds to see generalization.)")
    pickle.dump((regl,regr), open('config/daniel_right_mono_model_v02_and_v03.p','wb'))

    num_valid = int(len(X) / kfolds)
    num_train = len(X) - num_valid
    best_loss = np.float('inf')
    best_idx = -1
    avg_loss = 0.0

    for k in range(kfolds):
        vstart = k*num_valid
        vend = (k+1)*num_valid
        X_valid = X[vstart:vend]
        Y_valid = Y[vstart:vend]
        X_train = np.concatenate((X[:vstart], X[vend:]))
        Y_train = np.concatenate((Y[:vstart], Y[vend:]))
        print("\n  On kfold {}, valid range {} to {} (inclusive,exclusive)".format(k, vstart, vend))
        print("with shapes X_train, X_valid = {}, {}".format(X_train.shape, X_valid.shape))
        regl, lossl = train(X_train, Y_train[:,0:2], arm='left ', X_valid=X_valid, Y_valid=Y_valid[:,0:2], n_estimators=num_trees)
        regr, lossr = train(X_train, Y_train[:,2:4], arm='right', X_valid=X_valid, Y_valid=Y_valid[:,2:4], n_estimators=num_trees)
        loss = (lossl + lossr) / 2. # Will average losses among arms
        avg_loss += loss
        if loss < best_loss:
            best_loss = loss
            best_idx = k
    avg_loss /= kfolds
    print("\nbest_loss: {:.6f} at index {}, w/avg_loss: {:.6f}".format(best_loss, best_idx, avg_loss))
