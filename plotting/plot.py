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
titlesize = 30
labelsize = 25
legendsize = 25
ticksize = 25


def plot(data, ordered_keys):
    """ 
    Plotting ... note that the keys are really from `numbers.`
    """
    print("\nNow plotting!")
    colors = ['red', 'black', 'yellow', 'blue']
    plt.figure(figsize=(10,8))

    for i,key in enumerate(ordered_keys):
        data_mean = np.mean(data[key], axis=0)
        data_std  = np.std(data[key], axis=0) / np.sqrt(10) # If doing standard error...
        print("key {}, mean.shape {}".format(key,data_mean.shape))
        print("data_mean[{}][-1]: {}".format(key, data_mean[-1]))
        assert len(data_mean) == 400
        assert len(data_std) == 400
        plt.plot(np.arange(400), data_mean, 
                linewidth=3, 
                color=colors[i], 
                label="N={}".format(key))
        #plt.fill_between(np.arange(400),
        #        data_mean-data_std,
        #        data_mean+data_std,
        #        alpha=0.25,
        #        facecolor=colors[i])

    plt.title("Neural Network Validation Performance", fontsize=titlesize)
    plt.xlabel("Epochs", fontsize=labelsize)
    plt.ylabel("Validation MSE (Log Scale)", fontsize=labelsize)
    plt.yscale('log')
    plt.ylim([0.1,1000]) # Tweak
    plt.tick_params(axis='x', labelsize=ticksize)
    plt.tick_params(axis='y', labelsize=ticksize)
    plt.legend(loc='best', prop={'size':legendsize})
    plt.tight_layout()
    plt.savefig('neural_net_studies.png')


if __name__ == "__main__":
    """ 
    Load stuff. Remember, the data X_train is NOT normalized. 
    """
    num_train = 1552
    percentages = [0.25, 0.5, 0.75, 1.0]
    numbers = [int(pp*num_train) for pp in percentages]

    trial_stats = np.load('trial_stats.npy')[()]

    data = {} # What we use to plot in the figure
    per_mean = {} # For table (mean)
    per_std = {} # For table (std), but I think I'll ignore this. It's the
    # standard deviation for each of the ten trials and it doesn't seem to make
    # sense ... just list the average over the per_mean stuff.

    ordered_keys = sorted(trial_stats.keys())

    for key in ordered_keys:
        print("for key {}, the sub-dictionary has keys:\n\t{}".format(
            key, trial_stats[key].keys()))
        data[key] = np.array(
                [trial_stats[key]['val_loss_'+str(tt)] for tt in range(10)]
        )
        per_mean[key] = np.array(
                [trial_stats[key]['per_coord_val_errors_'+str(tt)] for tt in range(10)]
        )
        per_std[key] = np.array(
                [trial_stats[key]['per_coord_std_'+str(tt)] for tt in range(10)]
        )
        print("\ndata[{}].shape: {}".format(key, data[key].shape))
        print("per_mean[{}].shape: {}".format(key, per_mean[key].shape))
        print("per_std[{}].shape: {}".format(key, per_std[key].shape))

    for key in ordered_keys:
        mean_over_mean = np.mean(per_mean[key], axis=0)
        print("key {}, per-mean: {}".format(key, mean_over_mean))
        print(per_mean[key])
    plot(data, ordered_keys)
