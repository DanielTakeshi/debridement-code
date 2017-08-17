"""
Does the entire mapping pipeline.
"""

from dvrk.robot import *
from autolab.data_collector import DataCollector
from geometry_msgs.msg import PointStamped, Point
import image_geometry
import cv2
import numpy as np
import pickle
import sys
import time
np.set_printoptions(suppress=True)


# DOUBLE CHECK ALL THESE!!
LEFT_POINTS  = pickle.load(open('config/calib_circlegrid_left_v00_ONELIST.p',  'r'))
RIGHT_POINTS = pickle.load(open('config/calib_circlegrid_right_v00_ONELIST.p', 'r'))
C_LEFT_INFO  = pickle.load(open('config/camera_info_matrices/left.p',  'r'))
C_RIGHT_INFO = pickle.load(open('config/camera_info_matrices/right.p', 'r'))

ESC_KEYS     = [27, 1048603]


def initializeRobots():
    d = DataCollector()
    r1 = robot("PSM1") # left (but my right)
    r2 = robot("PSM2") # right (but my left)
    time.sleep(2)
    return (r1,r2,d)


def debug_1(left, right, points3d):
    print("\nSome debug prints:")
    for i,(pt1,pt2) in enumerate(zip(LEFT_POINTS, RIGHT_POINTS)):
        print(i,np.squeeze(pt1[0]),np.squeeze(pt2[0]))
    for i,(pt1,pt2) in enumerate(zip(left, right)):
        print(i,pt1,pt2)
    for i,item in enumerate(points_3d):
        print(i,np.squeeze(item))
    print("points_3d.shape: {}".format(points_3d.shape))
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


def solve_rigid_transform(camera_points_3d, robot_points_3d, debug=True):
    """
    Takes in two sets of corresponding points, returns the rigid transformation matrix 
    from the first to the second. This is a (3,4) matrix so we'd apply it on the original
    points in their homogeneous form with the fourth coordinate equal to one. This is
    slightly different from Brijen's code since I'm using np.array, not np.matrix.

    Notation: A for camera points, B for robot points, so want to find an affine mapping
    from A -> B with orthogonal rotation and a translation.
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
    t = meanA - R.dot(meanB)
    rigid_body_matrix = np.concatenate((R, t), axis=1)

    if debug:
        print("\nBegin debug prints:")
        print("meanA:\n{}\nmeanB:\n{}".format(meanA, meanB))
        print("Rotation R:\n{}\nand R^TR (should be identity):\n{}".format(R, (R.T).dot(R)))
        print("translation t:\n{}".format(t))
        print("rigid_body_matrix:\n{}".format(rigid_body_matrix))
        print("End of debug prints.\n")

    # Get residual to inspect quality of solution.

    return rigid_body_matrix


if __name__ == "__main__":
    arm1, _, d = initializeRobots()
    arm1.close_gripper()

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

    rigid_body_matrix = solve_rigid_transform(camera_points_3d=points_3d,
                                              robot_points_3d=robot_3d)
