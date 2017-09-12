""" A bar graph.

(c) September 2017 by Daniel Seita
"""

import argparse
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
titlesize = 21
labelsize = 17
legendsize = 15
ticksize = 15
bar_width = 0.80
opacity = 1.0
error_config = {'ecolor': '0.0', 'linewidth':3.0}


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


def plot(results, vv):
    lin_mean = []
    lin_std  = []
    lin_keys = []
    rfs_mean = []
    rfs_std  = []
    rfs_keys = []
    dnn_mean = []
    dnn_std  = []
    dnn_keys = []

    sorted_keys = sorted(results.keys())
    for key in sorted_keys:
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

    print("\nlin_mean: {}".format(lin_mean))
    print("lin_std:  {}".format(lin_std))
    print("lin_keys: {}".format(lin_keys))
    print("\nrfs_mean: {}".format(rfs_mean))
    print("rfs_std:  {}".format(rfs_std))
    print("rfs_keys: {}".format(rfs_keys))
    print("\nDNN results:")
    for (mean,std,key) in zip(dnn_mean,dnn_std,dnn_keys):
        print("{:.2f}\t{:.2f}\t{}".format(mean,std,key))
    # sys.exit()
    # Use this to determine which DNN models should be here.
    dnn_threshold = 3.0
    real_index = 0
    for ii,(mean,std,key) in enumerate(zip(dnn_mean,dnn_std,dnn_keys)):
        if mean > dnn_threshold:
            continue
        real_index += 1

    # Gah! Now I can finally make the bar chart. I think it's easiest to have it
    # split across three different subplots, one per algorithm category.
    width_ratio = [len(lin_keys),len(rfs_keys),real_index]
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(16,5),
                           gridspec_kw={'width_ratios':width_ratio})

    for ii,(mean,std,key) in enumerate(zip(lin_mean,lin_std,lin_keys)):
        ax[0].bar(np.array([ii]), mean, bar_width,
                  alpha=opacity,
                  yerr=std,
                  error_kw=error_config,
                  label=key[4:])
    for ii,(mean,std,key) in enumerate(zip(rfs_mean,rfs_std,rfs_keys)):
        ax[1].bar(np.array([ii]), mean, bar_width,
                  alpha=opacity,
                  yerr=std,
                  error_kw=error_config,
                  label=key[4:])
    real_index = 0
    for ii,(mean,std,key) in enumerate(zip(dnn_mean,dnn_std,dnn_keys)):
        if mean > dnn_threshold:
            continue
        ax[2].bar(np.array([real_index]), mean, bar_width,
                  alpha=opacity,
                  yerr=std,
                  error_kw=error_config,
                  label=key[4:])
        real_index += 1

    # Some rather tedious but necessary stuff to make it publication-quality.
    ax[0].set_title('Linear', fontsize=titlesize)
    ax[1].set_title('Random Forests', fontsize=titlesize)
    ax[2].set_title('Deep Neural Networks', fontsize=titlesize)
    ax[0].set_ylabel('Average Squared $L_2$, 10-Fold CV', fontsize=labelsize)

    for i in range(3):
        ax[i].set_xlabel('Algorithm', fontsize=labelsize)
        ax[i].set_ylim([0.0,9.0])
        ax[i].tick_params(axis='y', labelsize=ticksize)
        ax[i].set_xticklabels([])

    ax[0].legend(loc="best", ncol=1, prop={'size':legendsize})
    ax[1].legend(loc="best", ncol=2, prop={'size':legendsize})
    ax[2].legend(loc="best", ncol=3, prop={'size':legendsize})

    plt.tight_layout()
    plt.savefig('figures/validation_set_results_v'+vv+'.png')


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--version', type=int)
    pp.add_argument('--kfolds', type=int, default=10)
    args = pp.parse_args()
    assert args.version is not None

    VERSION = str(args.version).zfill(2)
    file_name = 'results/results_kfolds10_v'+VERSION+'.npy'
    results = np.load(file_name)[()]
    print("results has keys: {}".format(results.keys()))
    plot(results, VERSION)
