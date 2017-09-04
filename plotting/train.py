""" Easy to do ablation studies. :-) 

(c) September 2017 by Daniel Seita
"""

from collections import defaultdict
from keras.models import Sequential
from keras.layers import Dense, Activation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(suppress=True, linewidth=200)

# Some matplotlib settings.
plt.style.use('seaborn-darkgrid')
titlesize = 25
labelsize = 20
legendsize = 20
ticksize = 20


def train_network(X_train, y_train, X_valid, y_valid, X_mean, X_std):
    """ 
    Trains a network to predict f(cx,cy,cz,yaw) = (rx,ry,rz). Here, `X_train` is
    ALREADY normalized AND shuffled. X_train has shape (N,6), Y_train has shape
    (N,3). Thus, for keras, we use `input_dim = 6` since that is not including
    the batch size. Shuffle the data beforehand, since Keras will take the last
    % of the input data we pass as the fixed, held-out validation set.  
    
    DURING PREDICTION AND TEST-TIME APPLICATIONS, DON'T FORGET TO NORMALIZE!!!!!
    
    https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    """
    stats = defaultdict(list)
    N, input_dim = X_train.shape
    epochs = 400
    batch_size = 32

    model = Sequential()
    model.add(Dense(300, activation='relu', input_dim=input_dim))
    model.add(Dense(300, activation='relu')) 
    model.add(Dense(300, activation='relu'))
    model.add(Dense(3, activation=None))
    model.compile(optimizer='adam', loss='mse')

    # Use `history.history` to get `loss` and `val_loss` lists.
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(X_valid, y_valid))
    stats['loss'] = history.history['loss']
    stats['val_loss'] = history.history['val_loss']

    # Evaluate predictions
    y_valid_pred = model.predict(X_valid)
    ms_errors = 0.0
    num_valids = X_valid.shape[0]

    # This WILL overwrite ... this is mainly for debugging.
    with open("keras_results/numdata{}_epochs{}.txt".format(N, epochs), "w") as text_file:
        for index in range(num_valids):
            data = X_valid[index, :]
            targ = y_valid[index, :]
            pred = y_valid_pred[index, :]
            data = (data * X_std) + X_mean # Un-normalize the data!
            mse = np.linalg.norm(pred-targ) ** 2
            ms_errors += mse

            text_file.write("[{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}]   ===>   [{:.4f}, {:.4f}, {:.4f}],  actual: [{:.4f}, {:.4f}, {:.4f}] (mse: {:4f})\n".format(data[0],data[1],data[2],data[3],data[4],data[5], pred[0],pred[1],pred[2],targ[0],targ[1],targ[2], mse))
    print("mse as I computed it: {}".format(ms_errors / num_valids))

    # Collect additional stats for reporting in the ICRA 2018 submission.
    val_abs_errors = np.abs(y_valid_pred - y_valid)
    per_coord_val_errors = np.mean(val_abs_errors, axis=0)
    assert len(per_coord_val_errors) == 3
    stats['per_coord_val_errors'] = per_coord_val_errors
    print("per_coord_val_errors: {}".format(per_coord_val_errors))

    return stats


def plot(stats, numbers):
    """ 
    Plotting ... note that the keys are really from `numbers.`
    """
    colors = ['red', 'orange', 'black', 'yellow', 'blue']
    print("\nNow plotting with dictionary stats.keys(): {}".format(stats.keys()))
    print("Here are the per-coordinate errors for each run:")
    for nn in numbers:
        print("for key {}, per-coordinate errors: {}".format(nn, stats[nn]['per_coord_val_errors']))

    plt.figure(figsize=(10,8))
    for i,nn in enumerate(numbers):
        plt.plot(stats[nn]['val_loss'], linewidth=3, color=colors[i], label="N={}".format(nn))

    plt.title("Neural Network Validation Performance", fontsize=titlesize)
    plt.xlabel("Epochs", fontsize=labelsize)
    plt.ylabel("Validation MSE (Log Scale)", fontsize=labelsize)
    plt.yscale('log')
    #plt.ylim([...,...]) # Tweak
    plt.tick_params(axis='x', labelsize=ticksize)
    plt.tick_params(axis='y', labelsize=ticksize)
    plt.legend(loc='best', prop={'size':legendsize})
    plt.savefig('neural_net_studies.png')


if __name__ == "__main__":
    """ Load stuff. Remember, the data X_train is ALREADY normalized. """
    X_train = np.load("data/X_train.npy")
    X_mean  = np.load("data/X_mean.npy")
    X_std   = np.load("data/X_std.npy")
    y_train = np.load("data/y_train.npy")
    print("X_train.shape: {}".format(X_train.shape))
    print("y_train.shape: {}".format(y_train.shape))
    print("X_train:\n{}".format(X_train))
    print("y_train:\n{}".format(y_train))
    print("X_mean: {}".format(X_mean))
    print("X_std: {}".format(X_std))

    shuffle = np.random.permutation(X_train.shape[0])
    X_train = X_train[shuffle] # Already normalized!
    y_train = y_train[shuffle]
    N = X_train.shape[0]
    valid_split = 0.2    
    num_valid = int(N*valid_split)
    num_train = N - num_valid
    print("Number of validation points: {}".format(num_valid))
    print("Number of training points: {}".format(num_train))

    X_valid = X_train[:num_valid,:]
    y_valid = y_train[:num_valid,:]
    X_train = X_train[num_valid:,:]
    y_train = y_train[num_valid:,:]
    print("\nAfter getting validation split, we now have:")
    print("X_train.shape: {}".format(X_train.shape))
    print("y_train.shape: {}".format(y_train.shape))
    print("X_valid.shape: {}".format(X_valid.shape))
    print("y_valid.shape: {}".format(y_valid.shape))

    # Use this to determine how much "ablation"; percentages of full training data.
    percentages = [0.1, 0.25, 0.5, 0.75, 1.0]
    numbers = [int(pp*num_train) for pp in percentages]
    print("\nnumbers in training data: {}".format(numbers))
    trial_stats = {}

    for nn in numbers:
        # Use `_nn` to avoid overwriting `X_train` and `y_train`.
        indices = np.random.choice(num_train, size=nn, replace=False)
        X_train_nn = X_train[indices]
        y_train_nn = y_train[indices]
        print("\nUsing {} elements, X_train_nn/y_train_nn shapes ({},{})".format(
            nn, X_train_nn.shape, y_train_nn.shape))
        trial_stats[nn] = train_network(X_train_nn, y_train_nn, X_valid, y_valid, X_mean, X_std)

    plot(trial_stats, numbers)
