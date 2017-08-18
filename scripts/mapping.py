"""
Does the entire mapping pipeline. Saves stuff in ONE dictionary w/parameters.
"""

from dvrk.robot import *
from autolab.data_collector import DataCollector
from geometry_msgs.msg import PointStamped, Point
from sklearn.ensemble import RandomForestRegressor
import image_geometry
import cv2
import numpy as np
import pickle
import sys
import time
np.set_printoptions(suppress=True)

# DOUBLE CHECK ALL THESE!! Especially the version number!
VERSION      = '00'
LEFT_POINTS  = pickle.load(open('config/calib_circlegrid_left_v00_ONELIST.p',  'r'))
RIGHT_POINTS = pickle.load(open('config/calib_circlegrid_right_v00_ONELIST.p', 'r'))
C_LEFT_INFO  = pickle.load(open('config/camera_info_matrices/left.p',  'r'))
C_RIGHT_INFO = pickle.load(open('config/camera_info_matrices/right.p', 'r'))
NUM_POINTS   = len(LEFT_POINTS)
ESC_KEYS     = [27, 1048603]


def debug_1(left, right, points3d):
    print("\nSome debug prints:")
    print("robot points in left/right camera (we'll average later):")
    for i,(pt1,pt2) in enumerate(zip(LEFT_POINTS, RIGHT_POINTS)):
        print(i,np.squeeze(pt1[0]),np.squeeze(pt2[0]))
    print("pixel pts in left/right camera:")
    for i,(pt1,pt2) in enumerate(zip(left, right)):
        print(i,pt1,pt2)
    print("camera points:")
    for i,item in enumerate(points_3d):
        print(i,np.squeeze(item))
    print("(camera) points_3d.shape: {}".format(points_3d.shape))
    print("End of debug prints.\n")


def get_points_3d(left_points, right_points, info):
    """ 
    Assumes that corresponding (2-D pixel) points are ordered correctly 
    in `left_points` and `right_points`. Returns a list of 3D camera points.
    https://github.com/BerkeleyAutomation/endoscope_calibration/blob/master/rigid_transformation.py
    """
    points_3d = []
    for i in range(len(left_points)):
        a = left_points[i]
        b = right_points[i]
        disparity = np.sqrt((a[0]-b[0]) ** 2 + (a[1]-b[1]) ** 2)
        pt = convertStereo(a[0], a[1], disparity, info)
        points_3d.append(pt)
    return points_3d


def convertStereo(u, v, disparity, info):
    """ 
    Converts two pixel coordinates u and v with the disparity to give PointStamped.
    https://github.com/BerkeleyAutomation/endoscope_calibration/blob/master/rigid_transformation.py
    """
    stereoModel = image_geometry.StereoCameraModel()
    stereoModel.fromCameraInfo(info['l'], info['r'])
    (x,y,z) = stereoModel.projectPixelTo3d((u,v), disparity)
    cameraPoint = PointStamped()
    cameraPoint.header.frame_id = info['l'].header.frame_id
    cameraPoint.header.stamp = time.time()
    cameraPoint.point = Point(x,y,z)
    return cameraPoint


def pixels_to_3d():
    """ 
    Call this to start the process of getting camera points from pixels. 
    FYI, I explicitly extract the pixels from my points since they have more info.
    """
    left = []
    right = []
    info = {}

    # I _think_ ... hopefully this works. There isn't specific documentation.
    for (pt1, pt2) in zip(LEFT_POINTS, RIGHT_POINTS):
        _, _, lx, ly = pt1
        _, _, rx, ry = pt2
        left.append((lx,ly))
        right.append((rx,ry))
    info['l'] = C_LEFT_INFO
    info['r'] = C_RIGHT_INFO

    pts3d = get_points_3d(left, right, info)
    pts3d_array = np.asarray(np.matrix([(p.point.x, p.point.y, p.point.z) for p in pts3d]))
    return (left, right, pts3d_array)


def estimate_rf(X_train, Y_train, debug):
    """ Hopefully this will narrow down errors even further.
    
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
    has more error. But hopefully we can just lower it ... I hope.
    """
    rf = RandomForestRegressor(n_estimators=100)    
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_train)
    avg_l2_train = np.sum((Y_pred-Y_train)*(Y_pred-Y_train), axis=1)
    avg_l2 = np.mean(avg_l2_train)

    if debug:
        print("\nBegin debug prints for RFs:")
        print("X_train.T:\n{}".format(X_train.T))
        print("Y_train.T:\n{}".format(Y_train.T))
        print("Y_pred.T:\n{}".format(Y_pred.T))
        print("avg(|| ytarg-ypred ||_2^2) = {:.6f}".format(avg_l2))
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

    UPDATE: Also returns the random forest for estimating the residual noise!
    """
    assert camera_points_3d.shape == robot_points_3d.shape == (36,3)
    A = camera_points_3d.T # (3,N)
    B = robot_points_3d.T  # (3,N)

    # Look for Inge Soderkvist's solution online if confused.
    meanA = np.mean(A, axis=1, keepdims=True)
    meanB = np.mean(B, axis=1, keepdims=True)
    A = A - meanA
    B = B - meanB
    covariance = B.dot(A.T)
    U, sigma, VH = np.linalg.svd(covariance) # VH = V.T for our purposes.

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
    raw_errors = B_pred - B # Use pred-targ.
    l2_per_example = np.sum((B-B_pred)*(B-B_pred), axis=0)
    frobenius_loss = np.mean(l2_per_example)

    # Additional sanity checks. 
    print("\nCamera points (input), A.T:\n{}".format(A.T))
    print("Robot points (target), B.T:\n{}".format(B.T))
    print("Predicted robot points:\n{}".format(B_pred.T))
    print("Raw errors, B-B_pred:\n{}".format((B-B_pred).T))
    print("Residual (L2) for each:\n{}".format(l2_per_example.T))
    print("loss on data: {}".format(frobenius_loss))
    print("End of debug prints for rigid transformation.\n")

    # Now get that extra random forest. Actually we might have some liberty
    # with the input/target. For now, I use input=camera_pts, targs=abs_residual.
    X_train = camera_points_3d.T # (N,3)
    Y_train = raw_errors # (3,N)
    rf_residuals = estimate_rf(X_train, Y_train, debug)

    return RB_matrix, rf_residuals


def correspond_left_right_pixels(left, right, debug=True):
    """ This is my custom solution. To check it I should apply it on fixed
    points and just flip through the two images (literally click on the arrow
    keys) to see if they overlap.

    The solution is simple least squares. We can split this by column and it becomes
    minimizing L2 loss, i.e. ordinary linear regression, and solutions are known
    from the normal equations. But in case I'm paranoid I can just visualize solutions.

    Input: `left` and `right` are lists of (corresponding) tuples representing pixel values.
    """
    N = len(left)
    A = np.concatenate( (np.array(left),  np.ones((N,1))) , axis=1) # Left
    B = np.concatenate( (np.array(right), np.ones((N,1))) , axis=1) # Right
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

        cv2.imwrite('images/calib_left_w_left_points.jpg', img_left)
        cv2.imwrite('images/calib_right_w_left_points.jpg', img_right)

        # Now back to traditional debugs.
        print("\nBegin debug prints:")
        print("A.shape: {}, and A:\n{}".format(A.shape, A))
        print("B.shape: {}, and B:\n{}".format(B.shape, B))
        print("theta for LEFT to RIGHT:\n{}".format(theta_l2r))
        print("theta for RIGHT to LEFT:\n{}".format(theta_r2l))
        print("abs_errors l2r (remember, these are pixels):\n{}".format(abs_error_l2r))
        print("abs_errors r2l (remember, these are pixels):\n{}".format(abs_error_r2l))
        print("End of debug prints.\n")

    return theta_l2r, theta_r2l


if __name__ == "__main__":
    # Get the 3D **camera** points.
    assert len(LEFT_POINTS) == len(RIGHT_POINTS) == 36
    left, right, points_3d = pixels_to_3d()
    debug_1(left, right, points_3d)

    # Solve rigid transform. Average out the robot points (they should be similar).
    robot_3d = []
    for (pt1, pt2) in zip(LEFT_POINTS, RIGHT_POINTS):
        pos_l, _, _, _ = pt1
        pos_r, _, _, _ = pt2
        pos_l = np.squeeze(np.array(pos_l))
        pos_r = np.squeeze(np.array(pos_r))
        robot_3d.append( (pos_l+pos_r) / 2. )
    robot_3d = np.array(robot_3d)

    rigid_body_matrix, residuals_mapping = solve_rigid_transform(
            camera_points_3d=points_3d,
            robot_points_3d=robot_3d,
            debug=True)

    # Develop correspondence between left and right camera pixels. I assume in real
    # application, we'll only use the left camera at one time, but with this correspondence,
    # we effectively "pretend" that we know what the right camera is viewing.
    theta_l2r, theta_r2l = correspond_left_right_pixels(left, right)

    # Save various matrices. I also need a guide to _using_ them.
    params = {}
    params['RB_matrix'] = rigid_body_matrix
    params['rf_residuals'] = residuals_mapping
    params['theta_l2r'] = theta_l2r
    params['theta_r2l'] = theta_r2l
    pickle.dump(params, open('config/mapping_results/params_matrices_v'+VERSION+'.p', 'w'))
