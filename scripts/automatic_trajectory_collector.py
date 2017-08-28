"""
Hopefully the last major coding project for an ICRA 2018 submission. See the main method
for the hyperparameters I'm using. The main idea is to run this code and watch the robot
_automatically_ collect trajectory data.

Usage: the code should be run in a two-stage process. Run this the first time for some
manual intervention to determine the boundaries of the workspace and the height bounds.
The second time is when the real action begins and the trajectories can be run a lot.

This code saves as we go, not at the end, so that it's robust to cases when the dVRK might
fail (e.g. that MTM reading error we've been seeing a lot lately).

(c) August 2017 by Daniel Seita
"""

import cv2
import numpy as np
import os
import pickle
import sys
import utilities as utils
from autolab.data_collector import DataCollector
from dvrk.robot import *
np.set_printoptions(suppress=True)


class AutoTrajCollector:

    def __init__(self, args):
        self.d = args['d']
        self.arm = args['arm']
        self.num_trajs = args['num_trajs']
        self.rots_per_stoppage = args['rots_per_stoppage']
        self.z_offset = args['z_offset']
        self.interpolation_interval = args['interpolation_interval']

        self.info = pickle.load( open(args['guidelines_dir'], 'r') )
        self.z_alpha = self.info['z_alpha']
        self.z_beta  = self.info['z_beta']
        self.z_gamma = self.info['z_gamma']


    def _get_z_from_xy_values(self, x, y):
        """ We fit a plane. """
        return (self.z_alpha * x) + (self.z_beta * y) + self.z_gamma

    
    def _get_thresholded_image(self, image):
        """ Use this to get thresholded images from RGB (not BGR) `image`. 
       
        This should produce images that are easy enough for contour detection code 
        later, which should detect the center of the largest contour and deduce that
        as the approximate location of the end-effector.

        We assume the color target is red, FYI, and that it's RGB->HSV, not BGR->HSV.
        """
        # TODO we need extra stuff here, e.g. median blur and stuff, to reduce noise.
        lower = np.array([110, 90, 90])
        upper = np.array([180, 255, 255])
        hsv   = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask  = cv2.inRange(hsv, lower, upper)
        res   = cv2.bitwise_and(image, image, mask=mask)
        return res


    def collect_trajectories(self):
        """ Runs the robot and collects `self.num_trajs` trajectories. 
        
        1. Choose a random location of (x,y,z) that is within legal and safe range.
        2. Command the robot to move there, but slowly so that we can stop it along its trajectory.
            (TODO: I _think_ we can do this if we dive into the linear_interpolation method...)
        3. For each point in its trajectory, we'll randomly make it rotate to certain spots.
        4. Keep collecting and saving left AND right camera images of this process.
        5. Each trajectory is saved in its own sub-directory of images.

        We're going to save the normal camera views and the thresholded ones. The normal views
        are solely for debugging and reassurance, though we might do some systematic checking.

        TODO should also be careful about time sleeps and delays in taking images. 
        """
        traj_dirs = [x for x in os.listdir('traj_collector') if 'traj' in x]
        traj_index = len(traj_dirs)
        print("\nNow collecting trajectories. Starting index: {}\n".format(traj_index))
        
        for traj in range(self.num_trajs):
            this_dir = 'traj_collector/traj_'+str(traj_index).zfill(4)+'/'
            os.makedirs(this_dir)
            os.makedirs(this_dir+'left/')
            os.makedirs(this_dir+'right/')
            os.makedirs(this_dir+'left_th')
            os.makedirs(this_dir+'right_th/')
            intervals_in_traj = 0

            # Pick a safe target position. 
            # TODO figure out how to spread this more evenly across the workspace
            xx = np.random.uniform(low=self.info['min_x'], high=self.info['max_x'])
            yy = np.random.uniform(low=self.info['min_y'], high=self.info['max_y'])
            zz = self._get_z_from_xy_values(xx, yy)
            target_position = [xx, yy, zz + self.z_offset] 
            print("Trajectory {}, target position: {}".format(traj, target_position))

            # ------------------------------------------------------------------
            # Mostly following the `linear_interpolation` movement code.
            interval   = 0.0001
            start_vect = self.arm.get_current_cartesian_position()

            # Get a list with the yaw, pitch, and roll from the _starting_ position.
            _, current_rot = utils.lists_of_pos_rot_from_frame(start_vect)

            # If calling movement code, `end_vect` would be the input "tfx frame."
            end_vect     = tfx.pose(target_position, tfx.tb_angles(*current_rot))
            displacement = np.array(end_vect.position - start_vect.position)    

            # Total interval present between start and end poses
            tinterval = max(int(np.linalg.norm(displacement)/ interval), 50)    
            print("Number of intervals: {}".format(tinterval))

            for ii in range(0, tinterval, self.interpolation_interval):
                # SLERP interpolation from tfx function (from `dvrk/robot.py`).
                mid_pose = start_vect.interpolate(end_vect, (ii+1.0)/ tinterval)   
                arm.move_cartesian_frame(mid_pose, interpolate=True)
                print("interval {} of {}, mid_pose: {}".format(ii+1,tinterval,mid_pose))

                # --------------------------------------------------------------
                # Back to my own stuff, assuming that the arm is at `mid_pose`.
                frame = self.arm.get_current_cartesian_position()
                this_pos, this_rot = utils.lists_of_pos_rot_from_frame(frame)
                utils.move(arm=self.arm, pos=this_pos, rot=this_rot, SPEED_CLASS='Slow')
                time.sleep(3)

                # After moving there (keeping rotation fixed) we save the images.
                num = str(intervals_in_traj).zfill(3)
                left_thresholded  = self._get_thresholded_image(self.d.left_image.copy())
                right_thresholded = self._get_thresholded_image(self.d.right_image.copy())
                cv2.imwrite(this_dir+'left/'+num+'_rot0_left.jpg',        self.d.left_image)
                cv2.imwrite(this_dir+'right/'+num+'_rot0_right.jpg',      self.d.right_image)
                cv2.imwrite(this_dir+'left_th/'+num+'_rot0_left_th.jpg',  left_thresholded)
                cv2.imwrite(this_dir+'right_th/'+num+'_rot0_right_th.jpg', right_thresholded)
           
                for rr in range(1, self.rots_per_stoppage+1):
                    # Pick a random rotation and move there.
                    yaw   = np.random.uniform(low=self.info['min_yaw'],   high=self.info['max_yaw'])
                    pitch = np.random.uniform(low=self.info['min_pitch'], high=self.info['max_pitch'])
                    roll  = np.random.uniform(low=self.info['min_roll'],  high=self.info['max_roll'])
                    random_rotation = [yaw, pitch, roll]
                    utils.move(arm=self.arm, pos=this_pos, rot=random_rotation, SPEED_CLASS='Slow')
                    time.sleep(3)

                    # Save the left and right camera views (and the _thresholded_ ones).
                    rr = str(rr)
                    left_thresholded  = self._get_thresholded_image(self.d.left_image.copy())
                    right_thresholded = self._get_thresholded_image(self.d.right_image.copy())
                    cv2.imwrite(this_dir+'left/'+num+'_rot'+rr+'_left.jpg',         self.d.left_image)
                    cv2.imwrite(this_dir+'right/'+num+'_rot'+rr+'_right.jpg',       self.d.right_image)
                    cv2.imwrite(this_dir+'left_th/'+num+'_rot'+rr+'_left_th.jpg',   left_thresholded)
                    cv2.imwrite(this_dir+'right_th/'+num+'_rot'+rr+'_right_th.jpg', right_thresholded)

                # Back to the original rotation; I think this will make it work better.
                utils.move(arm=self.arm, pos=this_pos, rot=this_rot, SPEED_CLASS='Slow')
                intervals_in_traj += 1

            traj_index += 1


def collect_guidelines(arm, d, directory):
    """ 
    Gather statistics about the workspace on how safe we can set things. 
    Save things in a pickle file specified by the `directory` parameter. 
    Click the ESC key to exit the whole program and restart.
    """
    # First, add stuff that we should already know, particularly the rotation ranges.
    info = {}
    info['min_yaw']   = -90
    info['max_yaw']   =  90
    info['min_pitch'] = -50
    info['max_pitch'] =  50
    info['min_roll']  = -180
    info['max_roll']  = -100

    # Move the arm to positions to determine approximately safe ranges for x,y,z values.
    # And to be clear, all the `pos_{lr,ll,ul,ur}` are in robot coordinates.
    utils.call_wait_key(cv2.imshow("Left camera (move to lower right corner now!)", d.left_image))
    pos_lr = arm.get_current_cartesian_position()
    utils.call_wait_key(cv2.imshow("Left camera (move to lower left corner now!)", d.left_image))
    pos_ll = arm.get_current_cartesian_position()
    utils.call_wait_key(cv2.imshow("Left camera (move to upper left corner now!)", d.left_image))
    pos_ul = arm.get_current_cartesian_position()
    utils.call_wait_key(cv2.imshow("Left camera (move to upper right corner now!)", d.left_image))
    pos_ur = arm.get_current_cartesian_position()
    
    # So P[:,0] is a vector of the x's, P[:,1] vector of y's, P[:,2] vector of z's.
    p_lr = np.squeeze(np.array( pos_lr.position[:3] ))
    p_ll = np.squeeze(np.array( pos_ll.position[:3] ))
    p_ul = np.squeeze(np.array( pos_ul.position[:3] ))
    p_ur = np.squeeze(np.array( pos_ur.position[:3] ))
    P = np.vstack((p_lr, p_ll, p_ul, p_ur))

    # Get ranges. This is a bit of a heuristic but generally good I think.
    info['min_x'] = np.min( [p_lr[0], p_ll[0], p_ul[0], p_ur[0]] )
    info['max_x'] = np.max( [p_lr[0], p_ll[0], p_ul[0], p_ur[0]] )
    info['min_y'] = np.min( [p_lr[1], p_ll[1], p_ul[1], p_ur[1]] )
    info['max_y'] = np.max( [p_lr[1], p_ll[1], p_ul[1], p_ur[1]] )

    # For z, we fit a plane. See https://stackoverflow.com/a/1400338/3287820
    # Find (alpha, beta, gamma) s.t. f(x,y) = alpha*x + beta*y + gamma = z.
    A = np.zeros((3,3)) # Must be symmetric!
    A[0,0] = np.sum(P[:,0] * P[:,0])
    A[0,1] = np.sum(P[:,0] * P[:,1])
    A[0,2] = np.sum(P[:,0])
    A[1,0] = np.sum(P[:,0] * P[:,1])
    A[1,1] = np.sum(P[:,1] * P[:,1])
    A[1,2] = np.sum(P[:,1])
    A[2,0] = np.sum(P[:,0])
    A[2,1] = np.sum(P[:,1])
    A[2,2] = P.shape[0]

    b = np.array(
            [np.sum(P[:,0] * P[:,2]), 
             np.sum(P[:,1] * P[:,2]), 
             np.sum(P[:,2])]
    )

    x = np.linalg.inv(A).dot(b)
    info['z_alpha'] = x[0]
    info['z_beta']  = x[1]
    info['z_gamma'] = x[2]

    # Sanity checks before saving stuff.
    assert info['min_x'] < info['max_x']
    assert info['min_y'] < info['max_y']
    assert P.shape == (4,3)
    for key in info:
        print("key,val = {},{}".format(key, info[key]))
    print("P:\n{}".format(P))
    print("A:\n{}".format(A))
    print("x:\n{}".format(x))
    print("b:\n{}".format(b))

    pickle.dump(info, open(directory, 'w'))


if __name__ == "__main__": 
    arm, _, d = utils.initializeRobots()
    arm.close_gripper()
    utils.move(arm=arm, pos=[0.0,0.06,-0.13], rot=[0,-10,-170], SPEED_CLASS='Slow')
    directory = 'traj_collector/guidelines.p'

    # We're going to be in one of two cases, as I already specified.
    if not os.path.isfile(directory):
        print("We're going to start the first step, to collect guidelines.")
        collect_guidelines(arm, d, directory)
    else:
        print("Guidelines exist. Now let's proceed to the automatic trajectory collector.")

        # Arguments. No argparse since we're not running this many times.
        args = {}
        args['d'] = d
        args['arm'] = arm
        args['guidelines_dir'] = directory
        args['num_trajs'] = 1
        args['rots_per_stoppage'] = 0
        args['z_offset'] = 0.001
        args['interpolation_interval'] = 20

        # Build the ATC and then collect trajectories!
        ATC = AutoTrajCollector(args)
        ATC.collect_trajectories()
