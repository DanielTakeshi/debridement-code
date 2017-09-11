""" A bar graph.

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
    plt.savefig('figures/neural_net_studies.png')


if __name__ == "__main__":
    results = np.load('results.npy')[()]
