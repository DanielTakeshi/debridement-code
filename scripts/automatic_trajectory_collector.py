"""
Hopefully the last major coding project for an ICRA 2018 submission. See the main method
for the hyperparameters I'm using. The main idea is to run this code and watch the robot
_automatically_ collect trajectory data.

Usage: the code should be run in a two-stage process. Run this the first time for some
manual intervention to determine the boundaries of the workspace and the height bounds.
The second time is when the real action begins and the trajectories can be run a lot.

This code saves as we go, not at the end, so that it's robust to cases when the dVRK might
fail (e.g. that MTM reading error we've been seeing a lot lately).

(c) September 2017 by Daniel Seita
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
        self.interpolation_interval = args['interpolation_interval']
        self.require_negative_roll = args['require_negative_roll']

        self.info = pickle.load( open(args['guidelines_dir'], 'r') )
        self.z_alpha = self.info['z_alpha']
        self.z_beta  = self.info['z_beta']
        self.z_gamma = self.info['z_gamma']

        self.z_offset = args['z_offset']
        self.z_offset_home = args['z_offset_home']
        self.home_pos_list = [    
                self.info['home_lr'],
                self.info['home_ll'],
                self.info['home_ul'],
                self.info['home_ur']
        ]
        print("\nHere's what we have in our `self.info`:")
        for key in self.info:
            print("{} ==> {}".format(key, self.info[key]))
        print("")
 

    def _get_z_from_xy_values(self, x, y):
        """ We fit a plane. """
        return (self.z_alpha * x) + (self.z_beta * y) + self.z_gamma

    
    def _get_thresholded_image(self, image, left):
        """ Use this to get thresholded images from RGB (not BGR) `image`. 
       
        This should produce images that are easy enough for contour detection code 
        later, which should detect the center of the largest contour and deduce that
        as the approximate location of the end-effector.

        We assume the color target is red, FYI, and that it's RGB->HSV, not BGR->HSV.
        """
        # Requires some tuning and inspecting stuff from the two cameras.
        if left:
            lower = np.array([110, 90, 90])
            upper = np.array([180, 255, 255])
        else:
            lower = np.array([110, 80, 80])
            upper = np.array([180, 255, 255])

        image = cv2.medianBlur(image, 9)
        image = cv2.bilateralFilter(image, 7, 13, 13)

        hsv   = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask  = cv2.inRange(hsv, lower, upper)
        res   = cv2.bitwise_and(image, image, mask=mask)
        res   = cv2.medianBlur(res, 9)
        return res


    def _get_contour(self, image, left):
        """ 
        Given `image` in HSV format, get the contour center. This is the WRIST position
        of the robot w.r.t. camera pixels, as the end-effector is hard to work with.
        AND return the center of the contour!
        """
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR) 
        image_bgr = image.copy()

        # Detect contours *inside* the bounding box (a heuristic).
        if left:
            xx, yy, ww, hh = self.d.get_left_bounds()
        else:
            xx, yy, ww, hh = self.d.get_right_bounds()

        # Note that contour detection requires a single channel image.
        (cnts, _) = cv2.findContours(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 
                                     cv2.RETR_TREE, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        contained_cnts = []

        # Find the centroids of the contours in _pixel_space_.
        for c in cnts:
            try:
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Enforce it to be within bounding box.
                if (xx < cX < xx+ww) and (yy < cY < yy+hh):
                    contained_cnts.append(c)
            except:
                pass

        # Go from `contained_cnts` to a target contour and process it. And to be clear, 
        # we're only using the LARGEST contour, which SHOULD be the colored tape!
        if len(contained_cnts) > 0:
            target_contour = sorted(contained_cnts, key=cv2.contourArea, reverse=True)[0] # Index 0
            try:
                M = cv2.moments(target_contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                peri = cv2.arcLength(target_contour, True)
                approx = cv2.approxPolyDP(target_contour, 0.02*peri, True)

                # Draw them on the `image_bgr` to return, for visualization purposes.
                cv2.circle(image_bgr, (cX,cY), 50, (0,0,255))
                cv2.drawContours(image_bgr, [approx], -1, (0,255,0), 3)
                cv2.putText(img=image_bgr, 
                            text="{},{}".format(cX,cY), 
                            org=(cX+10,cY+10), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1, 
                            color=(255,255,255), 
                            thickness=2)
            except:
                pass
            return image_bgr, (cX,cY)

        else:
            # Failed to find _any_ contour.
            return image_bgr, (-1,-1) 


    def _sample_yaw(self, old_yaw=None):
        """ Samples a valid yaw. """
        yaw = np.random.uniform(low=self.info['min_yaw'], high=self.info['max_yaw'])
        if old_yaw is not None:
            while (np.abs(yaw-old_yaw) > 150): # Prevents large extremes
                yaw = np.random.uniform(low=self.info['min_yaw'], high=self.info['max_yaw'])
        return yaw


    def _sample_pitch(self):
        """ Samples a valid pitch. """
        return np.random.uniform(low=self.info['min_pitch'], high=self.info['max_pitch'])


    def _sample_roll(self):
        """ Samples a valid roll. """
        if self.require_negative_roll:
            roll = np.random.uniform(low=self.info['min_roll'], high=self.info['max_roll']) 
        else:
            roll = np.random.uniform(low=-180, high=180) 
            while (self.info['roll_neg_ubound'] < roll < self.info['roll_pos_lbound']):
                roll = np.random.uniform(low=-180, high=180) 
        return roll


    def _move_to_random_home_position(self):
        """ Moves to one of the home positions we have set up. 
        
        One change: I adjust the angles to be in line with what we have. The reason is
        that we'll keep referring back to this angle when we move, and I want it to be
        inside the actual ranges of yaw, pitch, and roll we're considering. Hence, the
        `better_rotation` that's here.
        """
        pose = self.home_pos_list[ np.random.randint(len(self.home_pos_list)) ]
        home_pos, home_rot = utils.lists_of_pos_rot_from_frame(pose)
        home_pos[2] += self.z_offset_home
        utils.move(arm=self.arm, pos=home_pos, rot=home_rot, SPEED_CLASS='Slow')

        better_rotation = [self._sample_yaw(), self._sample_pitch(), self._sample_roll()]
        utils.move(arm=self.arm, pos=home_pos, rot=better_rotation, SPEED_CLASS='Slow')


    def _save_images(self, this_dir, num, rr):
        """ Saves me some typing. """
        left_th  = self._get_thresholded_image(self.d.left_image.copy(),  left=True)
        right_th = self._get_thresholded_image(self.d.right_image.copy(), left=False)
        left_th, l_center  = self._get_contour(left_th,  left=True)
        right_th, r_center = self._get_contour(right_th, left=False)

        cv2.imwrite(this_dir+'left/'+num+'_rot'+rr+'_left.jpg',         self.d.left_image)
        cv2.imwrite(this_dir+'right/'+num+'_rot'+rr+'_right.jpg',       self.d.right_image)
        cv2.imwrite(this_dir+'left_th/'+num+'_rot'+rr+'_left_th.jpg',   left_th)
        cv2.imwrite(this_dir+'right_th/'+num+'_rot'+rr+'_right_th.jpg', right_th)
        return l_center, r_center


    def collect_trajectories(self):
        """ Runs the robot and collects `self.num_trajs` trajectories. 
        
        1. Choose a random location of (x,y,z) that is within legal and safe range.
        2. Command the arm to move there, but stop it periodically along its trajectory.
        3. For each point in its trajectory, we'll randomly make it rotate to certain spots.
        4. Keep collecting and saving left AND right camera images of this process.
        5. Each trajectory is saved in its own sub-directory of images.

        We're going to save the normal camera views and the thresholded ones. The normal views
        are solely for debugging and reassurance, though we might do some systematic checking.

        I put in `time.sleep(...)` commands since there's a delay when camera images are updated.
        """
        traj_dirs = [x for x in os.listdir('traj_collector') if 'traj' in x]
        traj_index = len(traj_dirs)
        print("\nNow collecting trajectories. Starting index: {}\n".format(traj_index))
        
        for traj in range(self.num_trajs):

            # Get directories/stats set up, and move to a random home position w/correct angles.
            this_dir = 'traj_collector/traj_'+str(traj_index).zfill(4)+'/'
            os.makedirs(this_dir)
            os.makedirs(this_dir+'left/')
            os.makedirs(this_dir+'right/')
            os.makedirs(this_dir+'left_th')
            os.makedirs(this_dir+'right_th/')
            intervals_in_traj = 0
            traj_poses = []
            self._move_to_random_home_position()
            time.sleep(3)

            # Pick a safe target position. 
            xx = np.random.uniform(low=self.info['min_x'], high=self.info['max_x'])
            yy = np.random.uniform(low=self.info['min_y'], high=self.info['max_y'])
            zz = self._get_z_from_xy_values(xx, yy)
            target_position = [xx, yy, zz + self.z_offset] 
            print("\n\nTrajectory {}, target position: {}".format(traj, target_position))

            # ------------------------------------------------------------------
            # Follows the `linear_interpolation` movement code to take incremental sets.
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
                mid_pose = start_vect.interpolate(end_vect, (ii+1.0)/tinterval)   
                arm.move_cartesian_frame(mid_pose, interpolate=True)

                # --------------------------------------------------------------
                # Back to my stuff. Note that `frame` = `mid_pose` (in theory, at least).
                time.sleep(3)
                print("\ninterval {} of {}, mid_pose: {}".format(ii+1, tinterval, mid_pose))
                frame = self.arm.get_current_cartesian_position() # The _actual_ pose.
                this_pos, this_rot = utils.lists_of_pos_rot_from_frame(frame)
                old_yaw = this_rot[0]

                # After moving there (keeping rotation fixed) we record information.
                num = str(intervals_in_traj).zfill(3) # Increments by 1 each non-rotation movement.
                l_center, r_center = self._save_images(this_dir, num, '0')
                traj_poses.append( (frame,l_center,r_center) )
           
                for rr in range(1, self.rots_per_stoppage+1):
                    # Pick a random rotation and move there. Update `old_yaw` as well.
                    random_rotation = [self._sample_yaw(old_yaw), 
                                       self._sample_pitch(), 
                                       self._sample_roll()]
                    old_yaw = random_rotation[0]
                    utils.move(arm=self.arm, pos=this_pos, rot=random_rotation, SPEED_CLASS='Slow')
                    time.sleep(3)
                    frame = self.arm.get_current_cartesian_position()
                    random_rotation_str = ["%.2f" % v for v in random_rotation]
                    print("    rot {}, target rot:  {}".format(rr, random_rotation_str))
                    print("    rot {}, actual pose: {}".format(rr, frame))

                    # Record information.
                    l_center, r_center = self._save_images(this_dir, num, str(rr))
                    traj_poses.append( (frame,l_center,r_center) )

                # Back to the original rotation; I think this will make it work better.
                utils.move(arm=self.arm, pos=this_pos, rot=this_rot, SPEED_CLASS='Slow')
                intervals_in_traj += 1

            # Finished with this trajectory!
            print("Finished with trajectory. len(traj_poses): {}".format(len(traj_poses)))
            pickle.dump(traj_poses, open(this_dir+'traj_poses_list.p', 'w'))
            traj_index += 1


def collect_guidelines(arm, d, directory):
    """ 
    Gather statistics about the workspace on how safe we can set things. Save things in a 
    pickle file specified by the `directory` parameter. Click the ESC key to exit the 
    program and restart. BTW, the four poses we collect will be the four "home" positions 
    that I use later, though with more z-coordinate offset.

    Some information:

        `yaw` must be limited in [-180,180]  # But actually, [-90,90] is all we need.
        `pitch` must be limited in [-50,50]  # I _think_ ...
        `roll` must be limited in [-180,180] # I think ...

    Remember, if I change the numbers, it doesn't impact the code until re-building `guidelines.p`!!
    """
    # First, add stuff that we should already know, particularly the rotation ranges.
    info = {}
    info['min_yaw']   = -90
    info['max_yaw']   =  90
    info['min_pitch'] =  -5
    info['max_pitch'] =  15
    info['roll_neg_ubound'] = -150 # (-180, roll_neg_ubound)
    info['roll_pos_lbound'] =  150 # (roll_pos_lbound, 180)
    info['min_roll'] = -175
    info['max_roll'] = -160

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

    # Save these so that we can use them for `home` positions.
    info['home_lr'] = pos_lr
    info['home_ll'] = pos_ll
    info['home_ul'] = pos_ul
    info['home_ur'] = pos_ur
    
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
    """ 
    When running this for real, be sure to feed (i.e. `tee`) the output into a file 
    for my own future reference. Again, as I said earlier, we're in two cases, depending
    on whether a `guidelines.p` file already exists.

    For the second case, there's a bunch of arguments, but no argparse since this isn't
    being run many times in simulation. By the way, `z_offset` is for generic offsets to 
    avoid damaging the paper, but `z_offset_home` is only for the four `home` positions.
    """
    arm, _, d = utils.initializeRobots()
    arm.close_gripper()
    directory = 'traj_collector/guidelines.p'

    if not os.path.isfile(directory):
        print("We're going to start the first step, to collect guidelines.")
        utils.move(arm=arm, pos=[0.0,0.06,-0.13], rot=[0,-10,-170], SPEED_CLASS='Slow')
        collect_guidelines(arm, d, directory)
    else:
        print("Guidelines exist. Now let's proceed to the automatic trajectory collector.")
        args = {}
        args['d'] = d
        args['arm'] = arm
        args['guidelines_dir'] = directory
        args['z_offset'] = 0.002
        args['z_offset_home'] = 0.010
        args['interpolation_interval'] = 20
        args['require_negative_roll'] = True

        # THESE ARE THE MAIN VALUES TO CHANGE!!
        args['num_trajs'] = 5
        args['rots_per_stoppage'] = 3

        # Build the ATC and then collect trajectories!
        ATC = AutoTrajCollector(args)
        ATC.collect_trajectories()
