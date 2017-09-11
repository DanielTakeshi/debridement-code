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


def plot(results):
    lin_mean = []
    lin_std  = []
    lin_keys = []
    rfs_mean = []
    rfs_std  = []
    rfs_keys = []
    dnn_mean = []
    dnn_std  = []
    dnn_keys = []

    for key in results.keys():
        info = [ss['loss'] for ss in results[key]]
        if 'Lin' in key:
            lin_mean.append(np.mean(info))
            lin_std.append(np.std(info))
            lin_keys.append(key)
        elif 'RFs' in key:
            rfs_mean.append(np.mean(info))
            rfs_std.append(np.std(info))
            rfs_keys.append(key)
        elif 'DNN' in key:
            dnn_mean.append(np.mean(info))
            dnn_std.append(np.std(info))
            dnn_keys.append(key)

    print("\nmean_lin: {}".format(lin_mean))
    print("std_lin:  {}".format(lin_std))
    print("keys_lin: {}".format(lin_keys))
    print("\nmean_rfs: {}".format(rfs_mean))
    print("std_rfs:  {}".format(rfs_std))
    print("keys_rfs: {}".format(rfs_keys))
    print("\nmean_dnn: {}".format(dnn_mean))
    print("std_dnn:  {}".format(dnn_std))
    print("keys_dnn: {}".format(dnn_keys))

    # Gah! Now I can finally make the bar chart. I think it's easiest to have it
    # split across three different subplots, one per algorithm category.
    pass


def deprecated():
    """ 
    This is a deprecated method, only to show how to possibly combine these into
    one plot. However, I find this unwieldly.
    """
    fig, ax = plt.subplots()
    bar_width = 0.80
    opacity = 0.5
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(np.array([0,1]), means_lin, bar_width,
                     alpha=opacity,
                     color='b',
                     yerr=std_lin,
                     error_kw=error_config,
                     label='Lin')
    rects2 = plt.bar(np.array([3,4,5,6,7]), means_rfs, bar_width,
                     alpha=opacity,
                     color='r',
                     yerr=std_rfs,
                     error_kw=error_config,
                     label='RF')
    rects3 = plt.bar(np.array([9,10]), means_dnn, bar_width,
                     alpha=opacity,
                     color='y',
                     yerr=std_dnn,
                     error_kw=error_config,
                     label='DNN')

    plt.xticks(np.arange(11) + bar_width / 2, 
               ('A','B','','D','E','F','G','','','J','K'))
    plt.xlabel('Group')
    plt.ylabel('Scores')
    plt.title('Scores by group and gender')
    plt.tight_layout()
    plt.legend()
    plt.savefig('figures/validation_set_results.png')


if __name__ == "__main__":
    file_name = 'results/results_kfolds10_v00.npy'
    results = np.load(file_name)[()]
    print("results has keys: {}".format(results.keys()))
    plot(results)
