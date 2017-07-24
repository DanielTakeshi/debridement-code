"""
Given pickle files from the open loop for seeds, figure out the regressor.
Not totally sure how this will work but I think it's OK if we just do mappings from
(x1,y1) -> (x2,y2), where these are points in the robot space, not the pixel space.
Look at `load_open_loop_data` to see how I stored the data. OR we may need the original
pixels because in some cases I've seen vast differences. I think here we have two RFs
since we have one for rotation, other for movement.

Update: actually I want to try different things so let's put the camera stuff there as well.
"""

import environ
import pickle
import numpy as np
import sys
import tfx
from sklearn.ensemble import RandomForestRegressor
np.set_printoptions(suppress=True, linewidth=200)

#####################
# Change stuff here #
#####################
NUM_TREES = 10
DATA_FILE = 'data/demos_seeds_02.p'
OUT_FILE  = 'data/demos_seeds_02_four_mappings.p'
 

def load_open_loop_data(filename):
    """ Load data and manipulate it somehow and return the X, Y stuff. """
    data_rot = []
    data_xy  = []
    data_rot_camera = []
    data_xy_camera  = []
    f = open(filename,'r')

    while True:
        try:
            d = pickle.load(f)
            assert len(d) == 6 # I stored 4 seeds, but two required rotations, hence 4+2=6.
            for item in d:
                frame_before, frame_after, camera_pt, info = item

                if info == 'rotation':
                    # A bit tricky, but use `tfx.tb_angles(frame.rotation)` and extract individual angles.
                    rot_before = tfx.tb_angles(frame_before.rotation)
                    rot_after  = tfx.tb_angles(frame_after.rotation)
                    data_rot.append( 
                            [rot_before.yaw_deg, rot_before.pitch_deg, rot_before.roll_deg, 
                              rot_after.yaw_deg,  rot_after.pitch_deg,  rot_after.roll_deg]
                    )
                    data_rot_camera.append( 
                            [camera_pt[0], camera_pt[1], 
                             rot_before.yaw_deg, rot_before.pitch_deg, rot_before.roll_deg, 
                              rot_after.yaw_deg,  rot_after.pitch_deg,  rot_after.roll_deg]
                    )

                elif info == 'xy':
                    # Again, remember that we ignore the z-coordinate.
                    pos_before = list(frame_before.position)
                    pos_after  = list(frame_after.position)
                    assert len(pos_before) == len(pos_after) == 3
                    data_xy.append( 
                            [pos_before[0], pos_before[1], pos_after[0], pos_after[1]] 
                    )
                    data_xy_camera.append( 
                            [camera_pt[0], camera_pt[1], 
                             pos_before[0], pos_before[1], pos_after[0], pos_after[1]]
                    )
                else:
                    raise ValueError()
        except EOFError:
            break

    # Turn the above into correct numpy arrays. To make it easy, return in one dictionary.
    # For rotations: (y, p, r) -> (y', p', r') and for xy: (x, y) -> (x', y')
    # For the camera cases, the (cx,cy) points are concatenated with the input.
    data = {}
    data['X_rot']        = np.array(data_rot)[:, :3]
    data['Y_rot']        = np.array(data_rot)[:, 3:]
    data['X_xy']         = np.array(data_xy)[:, :2]
    data['Y_xy']         = np.array(data_xy)[:, 2:]
    data['X_rot_camera'] = np.array(data_rot_camera)[:, :5]
    data['Y_rot_camera'] = np.array(data_rot_camera)[:, 5:]
    data['X_xy_camera']  = np.array(data_xy_camera)[:, :4]
    data['Y_xy_camera']  = np.array(data_xy_camera)[:, 4:]
    print(data['X_rot'].shape)
    print(data['Y_rot'].shape)
    print(data['X_rot_camera'].shape)
    print(data['Y_rot_camera'].shape)
    return data


def train(X, Y):
    """ Yeah, just fits X,Y. Simple. RFs shouldn't overfit (too much, that is). """
    reg = RandomForestRegressor(n_estimators=NUM_TREES)
    reg.fit(X, Y)
    Y_pred = reg.predict(X)
    avg_l2_train = np.sum((Y_pred-Y)*(Y_pred-Y), axis=1)
    avg_l2 = np.mean(avg_l2_train)
    print("avg(|| ytarg-ypred ||_2^2) = {:.6f}".format(avg_l2))
    return reg


def visualize(rf):
    """ Given a random forest, figure out how to visualize this. Probably only works for xy. """
    pass


if __name__ == "__main__":
    data = load_open_loop_data(DATA_FILE)

    rf_rot        = train(data['X_rot'],        data['Y_rot'])
    rf_xy         = train(data['X_xy'],         data['Y_xy'])
    rf_rot_camera = train(data['X_rot_camera'], data['Y_rot_camera'])
    rf_xy_camera  = train(data['X_xy_camera'],  data['Y_xy_camera'])

    # Just save it all in once to reduce file clutter.
    pickle.dump((rf_rot, rf_xy, rf_rot_camera, rf_xy_camera), open(OUT_FILE,'wb'))

    # It would be great if we could visualize one of these!
    visualize(rf_xy)
