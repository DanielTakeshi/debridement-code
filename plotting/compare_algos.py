"""
Compares performance of different algorithms on the held-out validation set. I
should not be using that silly \'decrease dataset and try this\' stuff. Also,
don't do any plotting here. Save the data and then plot somewhere else.
"""

from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import numpy as np
import sys
np.set_printoptions(suppress=True, edgeitems=4, linewidth=200)


def convert_to_quaternion(X_train):
    """ Um ... is this a 3D --> 4D conversion? Yeah. I would just use this:

    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Source_Code

    And copy the formula (convert from C++ to Python).
    """
    assert X_train.shape[1] == 6
    X_train_q = np.zeros((X_train.shape[0],7)) # Have 7 columns...
    X_train_q[:,:3] = X_train[:,:3].copy()

    # The Wiki page does Yaw, Roll, then Pitch (so not Pitch, then Roll).
    cy = np.cos(X_train[:,3] * 0.5)
    sy = np.sin(X_train[:,3] * 0.5)
    cr = np.cos(X_train[:,5] * 0.5)
    sr = np.sin(X_train[:,5] * 0.5)
    cp = np.cos(X_train[:,4] * 0.5)
    sp = np.sin(X_train[:,4] * 0.5)

    qw = (cy * cr * cp) + (sy * sr * sp)
    qx = (cy * sr * cp) - (sy * cr * sp)
    qy = (cy * cr * sp) + (sy * sr * cp)
    qz = (sy * cr * cp) - (cy * sr * sp)

    X_train_q[:,3] = qw
    X_train_q[:,4] = qx
    X_train_q[:,5] = qy
    X_train_q[:,6] = qz
    return X_train_q


def load_data(kfolds=10, quaternion=False):
    """ Load the data for future usage. I call it `train` but it's really the
    full data.
    
    Do anything that requires adjusting the data (e.g., normalization) here.
    OH we should also do K-fold cross validation here. The kfolds will let me
    call `for train, test in kf.split(X):` later, where `train, test` are the
    indices for the training and testing, respectively.
    """
    if quaternion:
        X_train = convert_to_quaternion(np.load("data/X_train.npy"))
        X_mean  = np.mean(X_train, axis=0)
        X_std   = np.std(X_train, axis=0)
    else:
        X_train = np.load("data/X_train.npy")
        X_mean  = np.load("data/X_mean.npy")
        X_std   = np.load("data/X_std.npy")

    # In both cases, we want to _normalize_ the training data.
    print("X_train (NOT normalized):\n{}".format(X_train))
    X_train = (X_train - X_mean) / X_std
    y_train = np.load("data/y_train.npy")
    print("X_train.shape: {}".format(X_train.shape))
    print("y_train.shape: {}".format(y_train.shape))
    print("X_train (normalized):\n{}".format(X_train))
    print("y_train:\n{}".format(y_train))
    print("X_mean: {}".format(X_mean))
    print("X_std: {}".format(X_std))

    # Now form the k-folds. 
    kf = KFold(n_splits=kfolds)
    print("\nWe've formed the k-folds with kfolds: {}".format(kfolds))
    for ii,(train, test) in enumerate(kf.split(X_train)):
        print("fold {}, len(train)={}, len(test)={}".format(ii,len(train),len(test)))
    return X_train, y_train, kf


def get_per_coord_errors(y_valid, y_pred):
    """ Mean absolute errors, that is! """
    abs_err = np.abs(y_valid - y_pred)
    per_coord_err = np.mean(abs_err, axis=0)
    per_coord_std = np.std(abs_err, axis=0)
    assert len(per_coord_err) == 3 and len(per_coord_std) == 3
    return per_coord_err, per_coord_std


def get_loss(y_valid, y_pred):
    """ The loss (really MSE) should be averages over the N rows for this since
    the y_valid and y_pred are (N,dim), where dim=3 in our case.
    
    So we need one number for each ROW, and then we take the mean of that.
    However, each ROW is itself a MSE. That's why I have to take an average
    across rows.
    """
    abs_errors_sq = (y_pred-y_valid)*(y_pred-y_valid) # (N,3)
    mse_N = np.mean(abs_errors_sq, axis=1) # (N,)
    return np.mean(mse_N)


def lin(X_data, y_data, kf):
    """ A linear mapping.
    
    Should be worse than RFs and DNNs. Um, but it actually seems pretty good.
    Interesting...
    """
    print("\n\nNOW DOING: LIN\n")
    all_stats = []

    for tt,(train, test) in enumerate(kf.split(X_data)):
        X_train, X_test, y_train, y_test = \
                X_data[train], X_data[test], y_data[train], y_data[test]
        stats = defaultdict(list)

        regr = LinearRegression()
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        per_coord_err, per_coord_std = get_per_coord_errors(y_test, y_pred)

        stats['per_coord_err'] = per_coord_err
        stats['per_coord_std'] = per_coord_std
        stats['loss'] = get_loss(y_test, y_pred)
        print("(tt={}) final loss (mse): {:.4f}".format(tt, stats['loss']))
        all_stats.append(stats)

    print("mean over folds: {:.4f}".format(
        np.mean([stats['loss'] for stats in all_stats])))
    return all_stats


def rfs(X_data, y_data, kf, num_trees, max_depth=None):
    """ We might as well see what happens with random forests as the Phase I
    algorithm.
    
    Basically the usual bleh.
    """
    print("\n\nNOW DOING: RFs\n")
    print("num_trees: {}, max_depth: {}".format(num_trees, max_depth))
    all_stats = []

    for tt,(train, test) in enumerate(kf.split(X_data)):
        X_train, X_test, y_train, y_test = \
                X_data[train], X_data[test], y_data[train], y_data[test]
        stats = defaultdict(list)

        regr = RandomForestRegressor(n_estimators=num_trees, max_depth=max_depth)
        regr.fit(X_train, y_train)
        y_pred = regr.predict(X_test)
        per_coord_err, per_coord_std = get_per_coord_errors(y_test, y_pred)

        stats['per_coord_err'] = per_coord_err
        stats['per_coord_std'] = per_coord_std
        stats['loss'] = get_loss(y_test, y_pred)
        print("(tt={}) final loss (mse): {:.4f}".format(tt, stats['loss']))
        all_stats.append(stats)

    print("mean over folds: {:.4f}".format(
        np.mean([stats['loss'] for stats in all_stats])))
    return all_stats


def dnn(X_data, y_data, kf, num_units, num_hidden):
    """ Pray that this has best performance because then it's like I'm
    retroactively justifying my claims. :-)
    
    Uses a DNN with Keras. 
    """
    print("\n\nNOW DOING: DNN\n")
    print("num_hidden: {}, num_units: {}".format(num_hidden, num_units))
    all_stats = []
    batch_size = 32
    epochs = 100

    for tt,(train, test) in enumerate(kf.split(X_data)):
        X_train, X_test, y_train, y_test = \
                X_data[train], X_data[test], y_data[train], y_data[test]
        stats = defaultdict(list)

        model = Sequential()
        model.add(Dense(num_units, activation='relu', input_dim=6))
        for _ in range(num_hidden-1):
            model.add(Dense(num_units, activation='relu')) 
        model.add(Dense(3, activation=None))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        history = model.fit(X_train, y_train, 
                            verbose=0,
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_data=(X_test, y_test))
        y_pred = model.predict(X_test)
        per_coord_err, per_coord_std = get_per_coord_errors(y_test, y_pred)

        stats['per_coord_err'] = per_coord_err
        stats['per_coord_std'] = per_coord_std
        stats['loss'] = get_loss(y_test, y_pred)
        print("(tt={}) final loss (mse): {:.4f}".format(tt, stats['loss']))
        all_stats.append(stats)

    print("mean over folds: {:.4f}".format(
        np.mean([stats['loss'] for stats in all_stats])))
    return all_stats


if __name__ == "__main__":
    VERSION = 0
    kfolds = 10
    X_data, y_data, kf = load_data(kfolds)
    X_data_q, y_data_q, kf_q = load_data(kfolds, quaternion=True)
    results = {}

    # --------------------------------------------------------------------------
    # All these methods return a LIST _of_ dictionaries, so `results` is a
    # dictionary _of_ these lists. For each method, if `list` is what they
    # return, then `list[i]` contains statistics from trial i, where i is
    # zero-indexed.
    # --------------------------------------------------------------------------
    results['Lin_EA'] = lin(X_data, y_data, kf)
    results['Lin_Q']  = lin(X_data_q, y_data_q, kf_q)

    results['RFs_t10_dN']   = rfs(X_data, y_data, kf, num_trees=10, max_depth=None)
    results['RFs_t100_dN']  = rfs(X_data, y_data, kf, num_trees=100, max_depth=None)
    results['RFs_t1000_dN'] = rfs(X_data, y_data, kf, num_trees=1000, max_depth=None)
    results['RFs_t100_d10']  = rfs(X_data, y_data, kf, num_trees=100, max_depth=10)
    results['RFs_t100_d100'] = rfs(X_data, y_data, kf, num_trees=100, max_depth=100)

    results['DNN_u30_h2']  = dnn(X_data, y_data, kf, num_units=30, num_hidden=2)
    results['DNN_u30_h3']  = dnn(X_data, y_data, kf, num_units=30, num_hidden=3)
    results['DNN_u300_h2'] = dnn(X_data, y_data, kf, num_units=300, num_hidden=2)
    results['DNN_u300_h3'] = dnn(X_data, y_data, kf, num_units=300, num_hidden=3)

    name = "results/results_kfolds{}_v{}".format(kfolds,str(VERSION).zfill(2))
    np.save(name, results)
