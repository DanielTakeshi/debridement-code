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

    # Load the linear regression stuff.
    means_lin = (
            np.mean([ss['loss'] for ss in results['Lin']]),
            np.mean([ss['loss'] for ss in results['Lin_Q']])
    )
    std_lin = (
            np.std([ss['loss'] for ss in results['Lin']]),
            np.std([ss['loss'] for ss in results['Lin_Q']])
    )
    print("means_lin: {}".format(means_lin))
    print("std_lin: {}".format(std_lin))

    # Load the RF regression stuff.
    means_rfs = (
            np.mean([ss['loss'] for ss in results['RFs_t10_dN']]),
            np.mean([ss['loss'] for ss in results['RFs_t100_dN']]),
            np.mean([ss['loss'] for ss in results['RFs_t1000_dN']]),
            np.mean([ss['loss'] for ss in results['RFs_t100_d10']]),
            np.mean([ss['loss'] for ss in results['RFs_t100_d100']])
    )
    std_rfs = (
            np.std([ss['loss'] for ss in results['RFs_t10_dN']]),
            np.std([ss['loss'] for ss in results['RFs_t100_dN']]),
            np.std([ss['loss'] for ss in results['RFs_t1000_dN']]),
            np.std([ss['loss'] for ss in results['RFs_t100_d10']]),
            np.std([ss['loss'] for ss in results['RFs_t100_d100']])
    )
    print("means_rfs: {}".format(means_rfs))
    print("std_rfs: {}".format(std_rfs))

    # Almost done ... load the DNN regression stuff.
    means_dnn = (
            np.mean([ss['loss'] for ss in results['DNN_h30']]),
            np.mean([ss['loss'] for ss in results['DNN_h300']])
    )
    std_dnn = (
            np.std([ss['loss'] for ss in results['DNN_h30']]),
            np.std([ss['loss'] for ss in results['DNN_h300']])
    )
    print("means_dnn: {}".format(means_dnn))
    print("std_dnn: {}".format(std_dnn))

    # Gah! Now I can finally make the bar chart. I think it's easiest to have it
    # split across three different subplots, one per algorithm category.
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
