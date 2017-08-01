"""
Given pickle files from the human demonstrations for seeds, case 3, figure out
the regressor. There are several things which make this tricky. Main ideas:

- No rotations!

- We had eight seeds, and just picked them one by one. I stored demonstrations
  as `(frame_before_moving, frame_after_xy_move, pt_camera, 'xy')` tuples. Thus,
  when loading the data, each "item" in that pickle file is a LIST ... of EIGHT
  items, EACH of which is a tuple of FOUR elements. Got it? 

- (Then the number of items total is however many demonstrations we have.)

- With eight seeds, we should train eight random forests if we're doing
  behavioral cloning per time.

- We are not doing anything with regards to height. We are only going from
  (x1,y1) to (x2,y2), i.e. from robot space (where it thought to go) to another
  robot space (where I told it to go). My other code will just assign the
  'correct' height.

- Be careful, these eight are not always in the same order, i.e. they're not
  always the seeds in the right locations. I used eight seeds but their
  locations are fairly consistent so I want one random forest for "the first
  seed's rough location", another one for the "second seed's rough location" and
  so forth. Fortunately, the first two are definitely the first two seeds (in
  some order), then the next two, then the next two, and finally the last two.
  To detect which is which, just measure the height.

- And by the way, the actual numbers for the random forests should be indexed 0
  through 7. 
  
  - 0 is bottom right
  - 1 is top right
  - etc.
  - 6 is bottom left
  - 7 is top left

  So for all ambiguous pairs, make sure I understand, the one that is LOWER in
  the camera (which means having HIGHER y values for the pixels!!) should be
  part of the training data for the first random forest for that pair.

- Another option is to train one random forest across all eight times and see
  how that looks, to act as a pure behavioral cloning baseline.

(c) August 2017 by Daniel Seita
"""

import environ
import pickle
import numpy as np
import sys
#import tfx
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
np.set_printoptions(suppress=True, linewidth=200)

#####################
# Change stuff here #
#####################
NUM_TREES = 100
DATA_FILE = 'data/demos_seeds_03.p'
OUT_FILE  = 'data/demos_seeds_03_maps.p'
 

def load_open_loop_data(filename):
    """ Load data and manipulate it somehow and return the X, Y stuff. """
    data = defaultdict(list)
    f = open(filename,'r')

    while True:
        try:
            # This will load one demonstration, which consists of eight things.
            d = pickle.load(f)
            assert len(d) == 8

            # Remember, we have to iterate by two.
            for i in range(0,8,2):
                item0 = d[i]   # Ideally, bottom, but we don't know (yet).
                item1 = d[i+1] # Ideally, top, but again we don't know (yet).
                f_before_0, f_after_0, camera_0, _ = item0
                f_before_1, f_after_1, camera_1, _ = item1
                p_before_0 = list(f_before_0.position)
                p_after_0  = list(f_after_0.position)
                p_before_1 = list(f_before_1.position)
                p_after_1  = list(f_after_1.position)
                assert len(p_before_0) == len(p_before_1) == len(p_after_0) == len(p_after_1) == 3

                # Check the y component; `bottom` should be the lower one (when
                # looking at the camera image), with a larger y value. Sorry,
                # the naming can be a bit confusing. Draw it out on paper.
                bottom = str(i)
                top    = str(i+1)
                if camera_0[1] < camera_1[1]:
                    bottom = str(i+1)
                    top    = str(i)

                # Add the points to their respective lists. No z-coordinate!
                data['seed_'+bottom].append( 
                        [p_before_0[0], p_before_0[1], p_after_0[0], p_after_0[1]] 
                )
                data['seed_'+top].append( 
                        [p_before_1[0], p_before_1[1], p_after_1[0], p_after_1[1]] 
                )

                # And in addition, add both (no z-coordinate) to the generic list.
                data['all_seeds'].append( 
                        [p_before_0[0], p_before_0[1], p_after_0[0], p_after_0[1]] 
                )
                data['all_seeds'].append( 
                        [p_before_1[0], p_before_1[1], p_after_0[0], p_after_1[1]] 
                )

        except EOFError:
            break

    # Turn everything into numpy arrays. Then return the dictionary of lists!
    for key in data:
        data[key] = np.array(data[key])
    print("data[all_seeds].shape: {}".format(data['all_seeds'].shape))
    return data


def train(X, Y, key):
    """ Yeah, just fits X,Y. Simple. RFs shouldn't overfit (too much, that is). """
    reg = RandomForestRegressor(n_estimators=NUM_TREES)
    reg.fit(X, Y)
    Y_pred = reg.predict(X)
    avg_l2_train = np.sum((Y_pred-Y)*(Y_pred-Y), axis=1)
    avg_l2 = np.mean(avg_l2_train)
    print("{}, avg(|| ytarg-ypred ||_2^2) = {:.6f}".format(key, avg_l2))
    return reg


def visualize(rf):
    """ Given a random forest, figure out how to visualize this. TODO. """
    pass


if __name__ == "__main__":
    data = load_open_loop_data(DATA_FILE)

    forests = {}
    for key in data:
        assert len(data[key].shape) == 2 and data[key].shape[1] == 4
        X_train, Y_train = data[key][:2], data[key][2:]
        rf = train(X_train, Y_train, key=key)
        forests[key] = rf
    pickle.dump(forests, open(OUT_FILE, 'wb'))
