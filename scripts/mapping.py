"""
Does the entire mapping pipeline. Saves stuff in ONE dictionary w/parameters. Can 
handle both the manual and automatic calibration steps. I use argparse here so that 
I remember which settings to use and don't mistakenly use an outdated configuration.

Note that with the automatic collection, I should be dealing with rotations. And
that should _still_ use the rigid body transformation.

(c) September 2017 by Daniel Seita 
"""

from autolab.data_collector import DataCollector
from collections import defaultdict
from dvrk.robot import *
from keras.models import Sequential
from keras.layers import Dense, Activation
from geometry_msgs.msg import PointStamped, Point
from sklearn.ensemble import RandomForestRegressor
import argparse
import cv2
import image_geometry
import numpy as np
import pickle
import sys
import time
import utilities as utils
np.set_printoptions(suppress=True)
C_LEFT_INFO  = pickle.load(open('config/camera_info_matrices/left.p',  'r'))
C_RIGHT_INFO = pickle.load(open('config/camera_info_matrices/right.p', 'r'))


def train_network(X_train, Y_train, X_mean, X_std, version_out):
    """ 
    Trains a network to predict f(cx,cy,cz,yaw) = (rx,ry,rz). Here, `X_train` is 
    ALREADY normalized. X_train has shape (N,6), Y_train has shape (N,3). Thus, for
    keras, we use `input_dim = 6` since that is not including the batch size. Shuffle 
    the data beforehand, since Keras will take the last % of the input data we pass 
    as the fixed, held-out validation set.

    DURING PREDICTION AND TEST-TIME APPLICATIONS, DON'T FORGET TO NORMALIZE!!!!!

    https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
    """
    shuffle = np.random.permutation(X_train.shape[0])
    X_train = X_train[shuffle] # Already normalized!
    Y_train = Y_train[shuffle]

    N, input_dim = X_train.shape
    epochs = 400
    batch_size = 32
    val_split = 0.2

    model = Sequential()
    model.add(Dense(300, activation='relu', input_dim=input_dim))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(3,  activation=None))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_split=val_split)
    
    val_start = int(N*(1.-val_split))
    X_valid = X_train[val_start:, :]
    Y_valid = Y_train[val_start:, :]
    predictions = model.predict(X_valid)
    ms_errors = 0
    num_valids = X_valid.shape[0]
    head = "config/keras_results/"
    
    with open(head+"simple_{}epochs_v{}.txt".format(epochs,version_out), "w") as text_file:
        for index in range(num_valids):
            data = X_valid[index, :]
            targ = Y_valid[index, :]
            pred = predictions[index, :]
            data = (data * X_std) + X_mean   # Un-normalize the data!
            mse  = np.linalg.norm(pred-targ) ** 2
            ms_errors += mse

            text_file.write("[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]   ===>   [{:.3f}, {:.3f}, {:.3f}],  actual: [{:.3f}, {:.3f}, {:.3f}]  (mse: {:4f})\n".format(data[0],data[1],data[2],data[3],data[4],data[5], pred[0],pred[1],pred[2],targ[0],targ[1],targ[2], mse))
    
    print("X_mean: {}".format(X_mean))
    print("X_std: {}".format(X_std))
    print("val_start: {}".format(val_start))
    print("num_valids: {}".format(num_valids))
    print("mse as I computed it: {}".format(ms_errors / num_valids))

    modeldir = head+"simple_{}epochs_v{}.h5".format(epochs,version_out)
    model.save(modeldir)
    return modeldir


def debug_1(left, right, points3d, LEFT_POINTS, RIGHT_POINTS):
    print("\nSome debug prints:")
    print("robot points in left/right camera (we'll average later):")
    for i,(pt1,pt2) in enumerate(zip(LEFT_POINTS, RIGHT_POINTS)):
        print(i,np.squeeze(pt1[0]),np.squeeze(pt2[0]))
    #print("pixel pts in left/right camera:")
    #for i,(pt1,pt2) in enumerate(zip(left, right)):
    #    print(i,pt1,pt2)
    print("camera points:")
    for i,item in enumerate(points_3d):
        print(i,np.squeeze(item))
    print("(camera) points_3d.shape: {}".format(points_3d.shape))
    print("End of debug prints.\n")


def get_points_3d(left_points, right_points, info):
    """ 
    Assumes that corresponding (2-D pixel) points are ordered correctly 
    in `left_points` and `right_points`. Returns a numpy array of 3D camera points.
    https://github.com/BerkeleyAutomation/endoscope_calibration/blob/master/rigid_transformation.py
    """
    stereoModel = image_geometry.StereoCameraModel()
    stereoModel.fromCameraInfo(info['l'], info['r'])
    points_3d = []

    for i in range(len(left_points)):
        a = left_points[i]
        b = right_points[i]
        disparity = np.sqrt((a[0]-b[0]) ** 2 + (a[1]-b[1]) ** 2)
        assert disparity == np.linalg.norm(np.array(a)-np.array(b))
        (x,y,z) = stereoModel.projectPixelTo3d((a[0],a[1]), disparity)
        points_3d.append( [x,y,z] )

    return np.array(points_3d)


def pixels_to_3d(LEFT_POINTS, RIGHT_POINTS):
    """ 
    Call this to start the process of getting camera points from pixels. FYI, I
    explicitly extract the pixels from `{LEFT,RIGHT}_POINTS` since they have more info.
    """
    left = []
    right = []
    info = {'l':C_LEFT_INFO, 'r':C_RIGHT_INFO}

    # I _think_ ... hopefully this works. There isn't specific documentation.
    for (pt1, pt2) in zip(LEFT_POINTS, RIGHT_POINTS):
        _, _, lx, ly = pt1
        _, _, rx, ry = pt2
        left.append((lx,ly))
        right.append((rx,ry))

    pts3d_array = get_points_3d(left, right, info)
    return (left, right, pts3d_array)


def estimate_rf(X_train, Y_train, debug):
    """ Hopefully this will narrow down errors even further. UPDATE: this is the bad RF,
    but it's good to have a bad baseline anyway.
    
    X_train: the camera coordinates, array of (x,y,z) stuff, where z is computed via
            our method here which uses points from both cameras to get depth.
    Y_train: when we did the rigid transform, we get some error. These residuals are
            the targets, so (xp-xt,yp-yt,zp-zt) with pred-target for each component.
            These are raw values, so if we did the reverse we'd have to negate things.
            
    In real application, we need to get our camera points. Then, given those camera points,
    apply the rigid body to get the predicted robot frame point (xp,yp,zp), assuming fixed
    rotation for now. But, this has some error from that. The RF trained here can predict
    that, i.e. (xerr,yerr,zerr) = (xp-xt,yp-yt,zp-zt). Thus, we have to do:

        (xp,yp,zp) - (xerr,yerr,zerr)

    to get the new, more robust (hopefully) predictions. E.g. if xerr is 5mm, we keep over-
    shooting x by 5mm so better bring it down.

    Fortunately, this MIGHT not be needed, as I saw that our rigid body errors were 
    at most 2 millimeters in each of the x and y directions. Ironically, the z direction
    has more error. But hopefully we can just lower it ... I hope. The idea is that given
    the camera point, the RF should "know" how to correct for the rigid body.
    """
    assert X_train.shape[1] == Y_train.shape[1] == 3 # (N,3)
    rf = RandomForestRegressor(n_estimators=100)    
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_train)
    abs_errors = np.abs(Y_pred-Y_train)
    abs_mean_err = np.mean(abs_errors, axis=0)

    if debug:
        print("\nBegin debug prints for RFs:")
        print("X_train:\n{}".format(X_train))
        print("Y_train:\n{}".format(Y_train))
        print("Y_pred:\n{}".format(Y_pred))
        print("abs_mean_err: {}".format(abs_mean_err))
        print("End debug prints for RFs\n")
    return rf


def solve_rigid_transform(camera_points_3d, robot_points_3d, debug=True):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix 
    from the first to the second. This is a (3,4) matrix so we'd apply it on the original
    points in their homogeneous form with the fourth coordinate equal to one. This is
    slightly different from Brijen's code since I'm using np.array, not np.matrix.

    Notation: A for camera points, B for robot points, so want to find an affine mapping
    from A -> B with orthogonal rotation and a translation.

    UPDATE: Also returns the random forest for estimating the residual noise! This will be
    the "bad" RF baseline since the rigid body thinks it's already so good.
    """
    assert camera_points_3d.shape[1] == robot_points_3d.shape[1] == 3
    A = camera_points_3d.T # (3,N)
    B = robot_points_3d.T  # (3,N)

    # Look for Inge Soderkvist's solution online if confused.
    meanA = np.mean(A, axis=1, keepdims=True)
    meanB = np.mean(B, axis=1, keepdims=True)
    A = A - meanA
    B = B - meanB
    covariance = B.dot(A.T)
    U, sigma, VH = np.linalg.svd(covariance) # VH = V.T, i.e. numpy transposes it for us.

    V = VH.T
    D = np.eye(3)
    D[2,2] = np.linalg.det( U.dot(V.T) )
    R = U.dot(D).dot(V.T)
    t = meanB - R.dot(meanA)
    RB_matrix = np.concatenate((R, t), axis=1)

    #################
    # SANITY CHECKS #
    #################

    print("\nBegin debug prints for rigid transformation:")
    print("meanA:\n{}\nmeanB:\n{}".format(meanA, meanB))
    print("Rotation R:\n{}\nand R^TR (should be identity):\n{}".format(R, (R.T).dot(R)))
    print("translation t:\n{}".format(t))
    print("RB_matrix:\n{}".format(RB_matrix))

    # Get residual to inspect quality of solution. Use homogeneous coordinates for A.
    # Also, recall that we're dealing with (3,N) matrices, not (N,3).
    # In addition, we don't want to zero-mean for real applications.
    A = camera_points_3d.T # (3,N)
    B = robot_points_3d.T  # (3,N)

    ones_vec = np.ones((1, A.shape[1]))
    A_h = np.concatenate((A, ones_vec), axis=0)
    B_pred = RB_matrix.dot(A_h)
    assert B_pred.shape == B.shape

    # Careful! Use raw_errors for the RF, but it will depend on pred-targ or targ-pred.
    raw_errors = B_pred - B # Use pred-targ, of shape (3,N)
    l2_per_example = np.sum((B-B_pred)*(B-B_pred), axis=0)
    frobenius_loss = np.mean(l2_per_example)

    # Additional sanity checks. 
    print("\nCamera points (input), A.T:\n{}".format(A.T))
    print("Robot points (target), B.T:\n{}".format(B.T))
    print("Predicted robot points:\n{}".format(B_pred.T))
    print("Raw errors, B-B_pred:\n{}".format((B-B_pred).T))
    print("Mean abs error per dim: {}".format( (np.mean(np.abs(B-B_pred), axis=1))) )
    print("Residual (L2) for each:\n{}".format(l2_per_example.T))
    print("loss on data: {}".format(frobenius_loss))
    print("End of debug prints for rigid transformation.\n")

    # Now get that extra random forest. Actually we might have some liberty
    # with the input/target. For now, I use input=camera_pts, targs=abs_residual.
    X_train = camera_points_3d # (N,3)
    Y_train = raw_errors.T # (N,3)
    rf_residuals = estimate_rf(X_train, Y_train, debug)

    # NOW, finally, let's combine the two! Recall that A, Ah are our camera data.
    errors = rf_residuals.predict(X_train) # (N,3)
    assert errors.shape[1] == B_pred.shape[0] == 3
    assert (errors.T).shape == B_pred.shape
    B_preds_with_rf = B_pred - errors.T # (3,N)
    new_raw_errors = B_preds_with_rf - B # (3,N)
    avg_abs_error_old = np.mean(np.abs(raw_errors))
    avg_abs_error_new = np.mean(np.abs(new_raw_errors))

    print("\nWhat if we COMBINE the rigid body with the RF?:")
    print("Predictions, B_pred-errors transposed (Rigid+RF):\n{}".format(B_preds_with_rf.T))
    print("NEW raw errors:\n{}".format(new_raw_errors.T))
    print("avg abs err for RB:    {}".format(avg_abs_error_old))
    print("avg abs err for RB+RF: {}".format(avg_abs_error_new))
    print("Mean abs error per dim: {}".format( np.mean(np.abs(new_raw_errors), axis=1) ))
    print("End of debug prints for rigid transform PLUS RF...\n")

    return RB_matrix, rf_residuals


def correspond_left_right_pixels(left, right, collection, VERSION, debug=True):
    """ This is my custom solution. To check it I should apply it on fixed
    points and just flip through the two images (literally click on the arrow
    keys) to see if they overlap.

    The solution is simple least squares. We can split this by column and it becomes
    minimizing L2 loss, i.e. ordinary linear regression, and solutions are known
    from the normal equations. But in case I'm paranoid I can just visualize solutions.

    Input: `left` and `right` are lists of (corresponding) tuples representing pixel values.
    """
    N = len(left)
    A = np.concatenate( (np.array(left),  np.ones((N,1))) , axis=1) # Left pixels
    B = np.concatenate( (np.array(right), np.ones((N,1))) , axis=1) # Right pixels
    A_nobias = np.array(left)
    B_nobias = np.array(right)

    # Left to Right
    col0_l2r = (np.linalg.inv((A.T).dot(A)).dot(A.T)).dot(B[:,0])
    col1_l2r = (np.linalg.inv((A.T).dot(A)).dot(A.T)).dot(B[:,1])
    theta_l2r = np.column_stack((col0_l2r, col1_l2r))
    abs_error_l2r = np.abs(A.dot(theta_l2r) - B_nobias)

    # Right to Left
    col0_r2l = (np.linalg.inv((B.T).dot(B)).dot(B.T)).dot(A[:,0])
    col1_r2l = (np.linalg.inv((B.T).dot(B)).dot(B.T)).dot(A[:,1])
    theta_r2l = np.column_stack((col0_r2l, col1_r2l))
    abs_error_r2l = np.abs(B.dot(theta_r2l) - A_nobias)

    assert theta_l2r.shape == theta_r2l.shape == (3,2)

    if debug:
        # Also check the mapping on an actual image.
        img_left  = cv2.imread('camera_location/calibration_blank_image_left.jpg')
        img_right = cv2.imread('camera_location/calibration_blank_image_right.jpg')
        
        # The lists `left` and `right` have corresponding tuples. If we take (x,y) from
        # `left`, which is our default camera, say it points to the third circle in a grid.
        #  We want the matrices to provide us with the pixel points in the *right* camera
        #  that would overlap with the same conceptual area, i.e. the center of that grid.
        for (cX,cY) in left:
            cv2.circle(img_left, (cX,cY), 5, (255,0,0), thickness=-1)
            cv2.putText(img=img_left, text="{},{}".format(cX,cY), org=(cX,cY), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75, color=(255,0,0), thickness=2)

            pt = np.array([cX,cY,1])
            pt_right = pt.dot(theta_l2r)
            oX, oY = int(round(pt_right[0])), int(round(pt_right[1]))

            cv2.circle(img_right, (oX,oY), 5, (0,0,255), thickness=-1)
            cv2.putText(img=img_right, text="{},{}".format(oX,oY), org=(oX,oY), 
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.75, color=(0,0,255), thickness=2)

        if collection == 'auto':
            cv2.imwrite('images/calib_auto_left_w_left_points_v'+VERSION+'.jpg',  img_left)
            cv2.imwrite('images/calib_auto_right_w_left_points_v'+VERSION+'.jpg', img_right)
        elif collection == 'manual':
            cv2.imwrite('images/calib_manual_left_w_left_points_v'+VERSION+'.jpg',  img_left)
            cv2.imwrite('images/calib_manual_right_w_left_points_v'+VERSION+'.jpg', img_right)

        # Now back to traditional debugs.
        print("\nBegin debug prints:")
        print("(left pixels) A.shape: {}, and A:\n{}".format(A.shape, A))
        print("(right pixels) B.shape: {}, and B:\n{}".format(B.shape, B))
        print("theta for LEFT to RIGHT:\n{}".format(theta_l2r))
        print("theta for RIGHT to LEFT:\n{}".format(theta_r2l))
        print("abs_errors l2r (remember, these are pixels):\n{}".format(abs_error_l2r))
        print("abs_errors r2l (remember, these are pixels):\n{}".format(abs_error_r2l))
        print("End of debug prints.\n")

    return theta_l2r, theta_r2l


def left_pixel_to_robot_prediction(left, params, true_points):
    """ Similar to the method in `click_and_crop.py` except that we have a full list of
    points from the left camera.
    """
    pred_points = []

    for left_pt in left:
        leftx, lefty = left_pt
        left_pt_hom = np.array([leftx, lefty, 1.])
        right_pt = left_pt_hom.dot(params['theta_l2r'])

        # Copy the code I wrote to convert these pts to camera points.
        disparity = np.linalg.norm(left_pt-right_pt)
        stereoModel = image_geometry.StereoCameraModel()
        stereoModel.fromCameraInfo(C_LEFT_INFO, C_RIGHT_INFO)
        (xx,yy,zz) = stereoModel.projectPixelTo3d( (leftx,lefty), disparity )
        camera_pt = np.array([xx, yy, zz])

        # Now I can apply the rigid body and RF (if desired).
        camera_pt = np.concatenate( (camera_pt, np.ones(1)) )
        robot_pt = (params['RB_matrix']).dot(camera_pt)
        target = [robot_pt[0], robot_pt[1], robot_pt[2]]
        pred_points.append(target)

    pred_points = np.array(pred_points)

    print("\nAssuming we ONLY have the left pixels, we still predict the robot points.")
    print("pred points:\n{}".format(pred_points))
    if true_points is not None:
        abs_errors = np.abs(pred_points - true_points)
        abs_mean_errors = np.mean(abs_errors, axis=0)
        print("true points:\n{}".format(true_points))
        print("mean abs err: {}".format(abs_mean_errors))
        print("mean err: {}".format(np.mean(pred_points-true_points, axis=0)))
    print("Done w/debugging.\n")


if __name__ == "__main__":
    pp = argparse.ArgumentParser()
    pp.add_argument('--collection', type=str, help='auto or manual')
    pp.add_argument('--version_in', type=int, help='0X for gripper, 1X for scissors')
    pp.add_argument('--version_out', type=int, help='00 or 01')
    args = pp.parse_args()
    assert args.collection.lower() in ['auto', 'manual']
    assert args.version_in is not None
    assert args.version_out is not None

    # SAVE STUFF HERE!
    params = {}

    # Load left and right calibration data, since I saved them separately. That's only to 
    # maintain consistency w/the manual version (the auto one could be saved in one list).
    VERSION = str(args.version_in).zfill(2)
    if args.collection == 'auto':
        LEFT_POINTS  = pickle.load(open('config/grid_auto/auto_calib_left_'+VERSION+'.p',  'r'))
        RIGHT_POINTS = pickle.load(open('config/grid_auto/auto_calib_right_'+VERSION+'.p', 'r'))
    elif args.collection == 'manual':
        LEFT_POINTS  = pickle.load(open('config/grid/calib_circlegrid_left_v'+VERSION+'_ONELIST.p',  'r'))
        RIGHT_POINTS = pickle.load(open('config/grid/calib_circlegrid_right_v'+VERSION+'_ONELIST.p', 'r'))
        if VERSION == '00':
            assert len(LEFT_POINTS) == len(RIGHT_POINTS) == 36
        elif VERSION == '01' or VERSION == '10':
            assert len(LEFT_POINTS) == len(RIGHT_POINTS) == 35
    NUM_POINTS = len(LEFT_POINTS)

    # -----------------------------
    # Get the 3D **camera** points.
    # -----------------------------
    left, right, points_3d = pixels_to_3d(LEFT_POINTS, RIGHT_POINTS)
    debug_1(left, right, points_3d, LEFT_POINTS, RIGHT_POINTS)

    # ---------------------------------------------------------------------------
    # Gather data needed for future transformations, particulary robot points and
    # (for the auto collection) the rotations, for better calibration.
    # -----------------------------------------------------------------------------
    robot_3d = []
    rotations_3d = []

    for (pt1, pt2) in zip(LEFT_POINTS, RIGHT_POINTS):
        pos_l, rot_l, _, _ = pt1
        pos_r, rot_r, _, _ = pt2
        pos_l = np.squeeze(np.array(pos_l))
        pos_r = np.squeeze(np.array(pos_r))

        if args.collection == 'auto':
            assert np.allclose(pos_l, pos_r)
            assert np.allclose(rot_l, rot_r)
            robot_3d.append(pos_l)
            rotations_3d.append(rot_l)
        elif args.collection == 'manual':
            # We have two measurements per point.
            robot_3d.append( (pos_l+pos_r) / 2. ) 

    robot_3d     = np.array(robot_3d)
    rotations_3d = np.array(rotations_3d)
    print("robot_3d.shape:     {}".format(robot_3d.shape))
    print("rotations_3d.shape: {}".format(rotations_3d.shape))
    assert robot_3d.shape[0] == rotations_3d.shape[0]      # (N,3) for each!
    assert robot_3d.shape[1] == rotations_3d.shape[1] == 3 # (N,3) for each!
    assert len(robot_3d.shape) == len(rotations_3d.shape) == 2 

    # -------------------------------------------------------------------------
    # For manual, get rigid body transform. For auto we have to split by cases.
    # -------------------------------------------------------------------------
    yaw_stuff = []

    if args.collection == 'manual':
        rigid_body_matrix, residuals_mapping = solve_rigid_transform(
                camera_points_3d=points_3d,
                robot_points_3d=robot_3d,
                debug=True)

    elif args.collection == 'auto':
        yaws = [-90, -45, 0, 45, 90]    

        # For each yaw, determine indices that match it in its range.
        for yaw in yaws:
            print("\nCURRENTLY ON YAW = {}".format(yaw))
            y_min = yaw - 22.5
            y_max = yaw + 22.5
            indices = []

            # Extract rotation, and get the stored yaw from rot[0].
            for ii,rot in enumerate(rotations_3d):
                if y_min <= rot[0] < y_max:
                    indices.append(ii)

            # Using our indices, we now extract the appropriate stuff by indexing.
            indices = np.array(indices)
            this_rb_matrix, this_internal_rf = solve_rigid_transform(
                    camera_points_3d=points_3d[indices],
                    robot_points_3d=robot_3d[indices],
                    debug=False)
            params['rigid_body_'+str(yaw)]  = this_rb_matrix
            params['internal_rf_'+str(yaw)] = this_internal_rf
            yaw_stuff.append( (indices.shape, points_3d[indices].shape, robot_3d[indices].shape) )

    # Debugging ...
    nums = 0
    for item in yaw_stuff:
        print(item)
        nums += float(np.squeeze(item[0]))
    print("Total # of points: {}".format(nums))
    N = robot_3d.shape[0]
    assert N == nums

    # ------------------------------------------------------------------------------------
    # For auto, get lots of stuff, e.g. deep networks. 
    # For the rotations, keep only the 0th column (yaw). EDIT: nope, let's use pitch/roll!
    # Just remember that the input is (cx, cy, cz, yaw, pitch, roll), IN THAT ORDER!!
    # ------------------------------------------------------------------------------------
    if args.collection == 'auto':
        X_train = np.concatenate((points_3d, rotations_3d), axis=1) # (N,6)
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0)
        y_train = robot_3d.copy() # (N,3)
        assert X_train.shape[1] == 6 and y_train.shape[1] == 3 and X_train.shape[0] == y_train.shape[0]
        assert len(X_std) == len(X_mean) == 6
        X_train = (X_train - X_mean) / X_std
        modeldir = train_network(X_train, y_train, X_mean, X_std, args.version_out)

    # -----------------------------------------------------------------------------------
    # Develop correspondence between left and right camera pixels. I assume in real
    # application, we'll only use the left camera at one time, but w/this correspondence,
    # we effectively "pretend" that we know what the right camera is viewing.
    # Note: better to IGNORE this if using the automatic trajectory collector, since that
    # has much noiser left/right correspondences. Just use the manual one in application.
    # -----------------------------------------------------------------------------------
    theta_l2r, theta_r2l = correspond_left_right_pixels(left, right, args.collection, VERSION)

    # ----------------------------------------
    # Save various matrices in one dictionary. 
    # ----------------------------------------
    params['theta_l2r'] = theta_l2r
    params['theta_r2l'] = theta_r2l

    if args.collection == 'auto':
        params['X_mean'] = X_mean
        params['X_std']  = X_std
        params['modeldir'] = modeldir
        pickle.dump(params, 
                open('config/mapping_results/auto_params_matrices_v'+str(args.version_out).zfill(2)+'.p', 'w'))
        print("\nDictionary keys: {}".format(params.keys()))

    elif args.collection == 'manual':
        params['RB_matrix'] = rigid_body_matrix
        params['rf_residuals'] = residuals_mapping
        pickle.dump(params, 
                open('config/mapping_results/manual_params_matrices_v'+str(args.version_out).zfill(2)+'.p', 'w'))

    # -----------------------------------------------------------------------------
    # Let's see what happens if we _pretend_ we ignore the right camera's pixels.
    # Given the left camera's pixels, see if we can accurately predict robot frame.
    # -----------------------------------------------------------------------------
    #left_pixel_to_robot_prediction(left, params, robot_3d)
