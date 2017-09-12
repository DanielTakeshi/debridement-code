"""
Compares performance of different algorithms on the held-out validation set. I
should not be using that silly \'decrease dataset and try this\' stuff. Also,
don't do any plotting here. Save the data and then plot somewhere else.
"""

from collections import defaultdict
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Activation
from keras.models import Sequential
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import argparse
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
    """ MSE per coordinate. """
    per_coord_mse = np.mean((y_valid-y_pred)*(y_valid-y_pred), axis=0)
    per_coord_std =  np.std((y_valid-y_pred)*(y_valid-y_pred), axis=0)
    assert len(per_coord_mse) == 3 and len(per_coord_std) == 3
    return per_coord_mse, per_coord_std


def get_avg_sq_l2_dist(y_valid, y_pred):
    """ 
    The loss. Just get squared L2 distance between all the N elements. Then take
    an average. It's easier that way. Note that Keras actually computes a
    further average over the 3 elements but this is the same monotonically ...
    """
    abs_errors_sq = (y_pred-y_valid)*(y_pred-y_valid) # (N,3)
    squared_l2_distances_N = np.sum(abs_errors_sq, axis=1) # (N,)
    return np.mean(squared_l2_distances_N)


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
        stats['loss'] = get_avg_sq_l2_dist(y_test, y_pred)
        assert abs(np.sum(stats['per_coord_err']) - stats['loss']) < 0.0001
        print("(tt={}) final loss (avg sq l2): {:.4f}".format(tt, stats['loss']))
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
        stats['loss'] = get_avg_sq_l2_dist(y_test, y_pred)
        assert abs(np.sum(stats['per_coord_err']) - stats['loss']) < 0.0001
        print("(tt={}) final loss (avg sq l2): {:.4f}".format(tt, stats['loss']))
        all_stats.append(stats)

    print("mean over folds: {:.4f}".format(
        np.mean([stats['loss'] for stats in all_stats])))
    return all_stats


def dnn(X_data, y_data, kf, num_units, num_hidden, nonlin):
    """ Pray that this has best performance because then it's like I'm
    retroactively justifying my claims. :-)
    
    Uses a DNN with Keras. 
    """
    print("\n\nNOW DOING: DNN\n")
    print("num_hidden: {}, num_units: {}, nonlin: {}".format(
        num_hidden, num_units, nonlin))
    all_stats = []
    batch_size = 64
    epochs = 5000

    for tt,(train, test) in enumerate(kf.split(X_data)):
        X_train, X_test, y_train, y_test = \
                X_data[train], X_data[test], y_data[train], y_data[test]
        stats = defaultdict(list)

        model = Sequential()
        model.add(Dense(num_units, activation=nonlin, input_dim=6))
        for _ in range(num_hidden-1):
            model.add(Dense(num_units, activation=nonlin)) 
        model.add(Dense(3, activation=None))
        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # Uses the validation set we provide, early stopping, etc.
        #early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(X_train, y_train, 
                            verbose=0,
                            epochs=epochs, 
                            batch_size=batch_size, 
                            #callbacks=[early_stopping],
                            validation_data=(X_test,y_test))
        y_pred = model.predict(X_test)
        per_coord_err, per_coord_std = get_per_coord_errors(y_test, y_pred)

        # Might as well collect all of the losses ...
        stats['val_loss'] = history.history['val_loss']
        stats['per_coord_err'] = per_coord_err
        stats['per_coord_std'] = per_coord_std
        stats['loss'] = get_avg_sq_l2_dist(y_test, y_pred)
        assert abs(np.sum(stats['per_coord_err']) - stats['loss']) < 0.0001
        print("(tt={}) final loss (avg sq l2): {:.4f}".format(tt, stats['loss']))
        all_stats.append(stats)

    print("mean over folds: {:.4f}".format(
        np.mean([stats['loss'] for stats in all_stats])))
    return all_stats


def do_johns_stuff(X_data, y_data, kf):
    for tt,(train, test) in enumerate(kf.split(X_data)):
        X_train, X_test, y_train, y_test = \
                X_data[train], X_data[test], y_data[train], y_data[test]
        model = Sequential()
        model.add(Dense(20, activation='sigmoid', input_dim=6))
        model.add(Dense(40, activation='sigmoid')) 
        model.add(Dense(60, activation='sigmoid')) 
        model.add(Dense(40, activation='sigmoid')) 
        model.add(Dense(3, activation=None))
        model.compile(optimizer='adam', loss='mse')
        model.summary()
        history = model.fit(X_train, y_train, 
                            verbose=1,
                            epochs=5000, 
                            batch_size=64, 
                            validation_data=(X_test,y_test))
        y_pred = model.predict(X_test)
        avg_sq_l2 = get_avg_sq_l2_dist(y_test, y_pred)
        print("(tt={}) final loss (avg sq l2): {:.4f}".format(tt, avg_sq_l2))
        per_coord_err = np.mean( (y_pred-y_test)**2, axis=0 )
        print("mse per coord: {}".format(per_coord_err))
        print("np.sum(per_coord_err): {}".format(np.sum(per_coord_err)))
        sys.exit()


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--version', type=int)
    pp.add_argument('--kfolds', type=int, default=10)
    args = pp.parse_args()
    assert args.version is not None

    VERSION = str(args.version).zfill(2)
    X_data, y_data, kf = load_data(args.kfolds)
    X_data_q, y_data_q, kf_q = load_data(args.kfolds, quaternion=True)
    results = {}

    # Do John's stuff.
    #do_johns_stuff(X_data, y_data, kf)

    # --------------------------------------------------------------------------
    # All these methods return a LIST _of_ dictionaries, so `results` is a
    # dictionary _of_ these lists. For each method, if `list` is what they
    # return, then `list[i]` contains statistics from trial i, where i is
    # zero-indexed.
    # --------------------------------------------------------------------------
    results['Lin_EA'] = lin(X_data, y_data, kf)
    results['Lin_Q']  = lin(X_data_q, y_data_q, kf_q)

    results['RFs_t10_dN']    = rfs(X_data, y_data, kf, num_trees=10, max_depth=None)
    results['RFs_t100_dN']   = rfs(X_data, y_data, kf, num_trees=100, max_depth=None)
    results['RFs_t1000_dN']  = rfs(X_data, y_data, kf, num_trees=1000, max_depth=None)
    results['RFs_t1000_d10'] = rfs(X_data, y_data, kf, num_trees=1000, max_depth=10)
    results['RFs_t100_d10']  = rfs(X_data, y_data, kf, num_trees=100, max_depth=10)
    results['RFs_t100_d100'] = rfs(X_data, y_data, kf, num_trees=100, max_depth=100)

    # For the neural network, we'll benchmark against a LOT of possibilities!
    l_nonlin = ['sigmoid', 'tanh', 'relu']
    l_units = [30, 300]
    l_hlayers = [1, 2, 3]

    for units in l_units:
        for hlayers in l_hlayers:
            for nonlin in l_nonlin:
                key = "DNN_u{}_h{}_{}".format(units, hlayers, nonlin)
                results[key] = dnn(X_data, y_data, kf, 
                                   num_units=units, 
                                   num_hidden=hlayers,
                                   nonlin=nonlin)

    name = "results/results_kfolds{}_v{}".format(args.kfolds, VERSION)
    np.save(name, results)
