"""
Test with rotations here. Observations:

0. `arm.move_cartesian_rotation(rot_0)` needs PyKDL data structure, not sure how that works.

1. The "roll" to me actually looks like "yaw" if we view the long arm as an airplane.

2. Fortunately, "pitch" has the intuitive interpretation, and "yaw" looks like the "roll." Yeah...

3. Angle limits:
    yaw:   [-180, 180] # I think this has the full range of 360 degrees of motion. Good!
    pitch: [-50, 50] # Limited, intuitively like moving wrist up and down.
    roll:  [-180, -100] # Limited, intuitively like moving a wrist sideways, as in "yaw" I know...

4. Will have to find good configurations for rotations. I think yaw of -120, 0, and 120 is good 
since that covers the 360 degree circle uniformly and doesn't duplicate itself. So ... just find
empiricaly good pitch and roll for that? Ideally those make it easier to grasp stuff.

Edit: huh maybe just keeping pitch at -10 and roll at -170 is fine.
For yaw=90 maybe have pitch=0, roll=-165? To test, just put some seeds out at 90 degrees
and see how the pitch and roll looks.
"""

from autolab.data_collector import DataCollector
from dvrk.robot import *
import cv2
import image_geometry
import numpy as np
import os
import pickle
import sys
import tfx
import utilities as utils
np.set_printoptions(suppress=True)

###########################
# ADJUST THESE AS NEEDED! #
###########################

VERSION_INPUT  = '00'    # 00=gripper, 10=scissors
VERSION_OUTPUT = '99'    # See README in `images/` for details on numbers, e.g. use 99 for debugging.

# Adjust carefully!!
CLOSE_ANGLE    = 25      # I think 25 is good for pumpkin, 10 for sunflower.
TOPK_CONTOURS  = 1       # I usually do 8, for 8 seeds.
INTERPOLATE    = True    # We thought `False` would be faster but alas it doesn't even go to the locations.
SPEED_CLASS    = 'Fast'  # See `measure_speeds.py` for details.

# Loading stuff.
#RF_REGRESSOR = pickle.load(open('config/mapping_results/random_forest_predictor_v'+VERSION_INPUT+'.p', 'r'))
#PARAMETERS   = pickle.load(open('config/mapping_results/manual_params_matrices_v'+VERSION_INPUT+'.p', 'r'))

# Rotations and positions.
# [   0.07827103  -12.19706825 -169.63700296]
#HOME_POS     = [0.00, 0.06, -0.13]
#ROTATION     = utils.get_average_rotation(VERSION_INPUT)
#TFX_ROTATION = tfx.tb_angles(ROTATION[0], ROTATION[1], ROTATION[2])

# Saving stuff
IMAGE_DIR = 'images/seeds_v'+VERSION_OUTPUT

# Offsets, some heuristic (e.g. x-coord), some (e.g. the z-coord) for safety.
ARM1_XOFFSET   = 0.000
ARM1_YOFFSET   = 0.000
ARM1_ZOFFSET   = 0.004
ZOFFSET_SAFETY = 0.003 # What I actually use in practice

##########################
# END OF `CONFIGURATION` #
##########################

def motion_planning(contours_by_size, img, arm, rotations):
    """ The open loop.
    
    Parameters
    ----------
    contours_by_size: [list]
        A list of contours, arranged from largest area to smallest area.
    img: [np.array]
        Image the camera sees, in BGR form (not RGB).
    arm: [dvrk arm]
        Represents the arm we're using for the DVRK.
    rotations:
    """
    print("Identified {} contours but will keep top {}.".format(len(contours_by_size), TOPK_CONTOURS))
    img_for_drawing = img.copy()
    contours = list(contours_by_size)
    cv2.drawContours(img_for_drawing, contours, -1, (0,255,0), 3)
    places_to_visit = []

    # Iterate and find centers. We'll make the robot move to these centers in a sequence.
    # Note that duplicate contours should be detected beforehand.
    for i,cnt in enumerate(contours):
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        places_to_visit.append((cX,cY))

    # Collect only top K places to visit and insert ordering preferences. I do right to left.
    places_to_visit = places_to_visit[:TOPK_CONTOURS]
    places_to_visit = sorted(places_to_visit, key=lambda x:x[0], reverse=True)

    # Number the places to visit in an image so I see them.
    for i,(cX,cY) in enumerate(places_to_visit):
        cv2.putText(img=img_for_drawing, text=str(i), org=(cX,cY), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, color=(0,0,0), thickness=2)
        cv2.circle(img=img_for_drawing, center=(cX,cY), radius=3, color=(0,0,255), thickness=-1)
        print(i,cX,cY)

    # Show image with contours + exact centers. Exit if it's not looking good. 
    # If good, SAVE IT!!! This protects me in case of poor perception causing problems.
    trial = len([fname for fname in os.listdir(IMAGE_DIR) if '.png' in fname])
    cv2.imshow("Top-K contours (exit if not looking good), trial index is {}".format(trial), img_for_drawing)
    utils.call_wait_key()
    cv2.imwrite(IMAGE_DIR+"/im_"+str(trial).zfill(3)+".png", img_for_drawing)
    cv2.destroyAllWindows()

    # Given points in `places_to_visit`, must figure out the robot points for each.
    robot_points_to_visit = []
    print("\nHere's where we'll be visiting (left_pixels, robot_point):")
    for left_pixels in places_to_visit:
        robot_points_to_visit.append(
                utils.left_pixel_to_robot_prediction(
                        left_pt=left_pixels,
                        params=PARAMETERS,
                        better_rf=RF_REGRESSOR,
                        ARM1_XOFFSET=ARM1_XOFFSET,
                        ARM1_YOFFSET=ARM1_YOFFSET,
                        ARM1_ZOFFSET=ARM1_ZOFFSET
                )
        )
        print("({}, {})\n".format(left_pixels, robot_points_to_visit[-1]))

    # With robot points, tell it to _move_ to these points. Apply vertical offsets here, FYI.
    # Using the `linear_interpolation` is slower and jerkier than just moving directly.
    arm.open_gripper(degree=90, time_sleep=2)

    for (robot_pt, rot) in zip(robot_points_to_visit, rotations):
        # Go to the point. Note: switch rotation **first**?
        pos = [robot_pt[0], robot_pt[1], robot_pt[2]+ZOFFSET_SAFETY]
        utils.move(arm, HOME_POS, rot, SPEED_CLASS) 
        utils.move(arm, pos,      rot, SPEED_CLASS)

        # Lower, close, and raise the end-effector.
        pos[2] -= ZOFFSET_SAFETY
        utils.move(arm, pos, rot, SPEED_CLASS)
        arm.open_gripper(degree=CLOSE_ANGLE, time_sleep=2) # Need >=2 seconds!
        pos[2] += ZOFFSET_SAFETY
        utils.move(arm, pos, rot, SPEED_CLASS)

        # Back to the home position.
        utils.move(arm, HOME_POS, rot, SPEED_CLASS)
        arm.open_gripper(degree=90, time_sleep=2) # Need >=2 seconds!


def get_single_contour_center(image, contours, return_contour=False):
    """ Assuming there is _one_ contour to return, we find it (return as a list).  """

    for i,(cX,cY,approx,peri) in enumerate(contours):  
        if utils.filter_point(cX,cY, xlower=700, xupper=1400, ylower=500, yupper=1000):
            continue
        cv2.circle(img=image, center=(cX,cY), radius=50, color=(0,0,255), thickness=1)
        cv2.circle(img=image, center=(cX,cY), radius=4, color=(0,0,255), thickness=-1)
        cv2.drawContours(image=image, contours=[approx], contourIdx=-1, color=(0,255,0), thickness=3)
        cv2.imshow("ESC if duplicate/undesired, other key to proceed", image)
        firstkey = cv2.waitKey(0) 

        # Return outright if we're only going to return one contour.
        if firstkey not in utils.ESC_KEYS:
            if return_contour:
                return [cX,cY,approx,peri]
            else:
                return [cX,cY]


def proof_of_concept_part_one(left_im, right_im, left_contours, right_contours, arm):
    """ 
    See my notes. I'm effectively trying to argue why my DL way with automatic trajectory
    collection will work. Maybe this will work as a screenshot or picture sequence in an
    eventual paper? I hope ...
    """
    center_left  = get_single_contour_center(left_im,  left_contours)
    center_right = get_single_contour_center(right_im, right_contours)
    cv2.destroyAllWindows()
    camera_pt = utils.camera_pixels_to_camera_coords(center_left, center_right)
    print("(left, right) = ({}, {})".format(center_left, center_right))
    print("(cx,cy,cz) = ({:.4f}, {:.4f}, {:.4f})".format(*camera_pt))

    # Get rotation set to -90 to start.
    pos, rot = utils.lists_of_pos_rot_from_frame(arm.get_current_cartesian_position())
    utils.move(arm, pos, [-90, 10, -170], 'Slow') 

    # Now do some manual movement.
    arm.open_gripper(90)
    utils.call_wait_key( cv2.imshow("(left image) move to point keeping angle roughly -90)", left_im) )
    pos, rot = utils.lists_of_pos_rot_from_frame(arm.get_current_cartesian_position())
    utils.move(arm, pos, [-90, 10, -170], 'Slow')  # Because we probably adjusted angle by mistake.
    pos2, rot2 = utils.lists_of_pos_rot_from_frame(arm.get_current_cartesian_position())

    # At this point, pos, rot should be the -90 degree angle version which can grip this.
    print("pos,rot -90 yaw:\n({:.4f}, {:.4f}, {:.4f}), ({:.4f}, {:.4f}, {:.4f})".format(
        pos2[0],pos2[1],pos2[2],rot2[0],rot2[1],rot2[2]))

    # Next, let's MANUALLY move to the reverse angle, +90. This SHOULD use a different position.
    # This takes some practice to figure out a good location. Also, the reason for manual movement
    # is that when I told the code to go to +90 yaw, it was going to +180 for some reason ...
    utils.call_wait_key( cv2.imshow("(left image) move to +90 where it can grip seed)", left_im) )
    pos3, rot3 = utils.lists_of_pos_rot_from_frame(arm.get_current_cartesian_position())
    print("pos,rot after manual +90 yaw change:\n({:.4f}, {:.4f}, {:.4f}), ({:.4f}, {:.4f}, {:.4f})".format(pos3[0],pos3[1],pos3[2],rot3[0],rot3[1],rot3[2]))

    # Automatic correction in case we made slight error.
    utils.move(arm, pos3, [90, 0, -165], 'Slow') 
    pos4, rot4 = utils.lists_of_pos_rot_from_frame(arm.get_current_cartesian_position())

    # Now pos, rot should be the +90 degree angle version which can grip this.
    print("pos,rot +90 yaw:\n({:.4f}, {:.4f}, {:.4f}), ({:.4f}, {:.4f}, {:.4f})".format(
        pos4[0],pos4[1],pos4[2],rot4[0],rot4[1],rot4[2]))

    # And thus, this is case 1. SAME CAMERA POINT since we haven't moved the seed at all.
    # But this means we don't have the same ROBOT point that can grip the seed!!


def proof_of_concept_part_two(left_im, right_im, left_contours, right_contours, arm):
    """  Second part ... """
    center_left  = get_single_contour_center(left_im,  left_contours)
    center_right = get_single_contour_center(right_im, right_contours)
    cv2.destroyAllWindows()
    camera_pt = utils.camera_pixels_to_camera_coords(center_left, center_right)
    print("(left, right) = ({}, {})".format(center_left, center_right))
    print("(cx,cy,cz) = ({:.4f}, {:.4f}, {:.4f})".format(*camera_pt))


def get_contours(cnts, img):
    for c in cnts:
        try:
            # Approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # Find the centroids of the contours in _pixel_space_. :)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if utils.filter_point(cX,cY, xlower=700, xupper=1300, ylower=500, yupper=800):
                continue

            # Now fit an ellipse!
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(img, ellipse, (0,255,0), 2)
            img_copy = img.copy()
            cv2.imshow("Is this ellipse good? ESC to skip it, else return it.", img_copy)
            firstkey = cv2.waitKey(0) 

            if firstkey not in utils.ESC_KEYS:
                return (cX, cY, peri, approx, ellipse)
        except:
            pass
    return None


def detect_seed_orientation(left_im, right_im, left_contours, right_contours, arm, d):
    """ Detect seed orientation! 
    
    For now I think I'm going to try and fit an ellipse. Start by going through the
    left and right cameras to find a _single_ seed contour. Then fit an ellipse, and
    find the orientation of the angle.
    """
    #cv2.imshow("left image proc", d.left_image_proc)
    #key = cv2.waitKey(0) 
    #cv2.imshow("right image proc", d.right_image_proc)
    #key = cv2.waitKey(0) 

    (l_cnts, _) = cv2.findContours(d.left_image_proc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (r_cnts, _) = cv2.findContours(d.right_image_proc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    l_cnt = get_contours(l_cnts, left_im.copy())
    r_cnt = get_contours(r_cnts, right_im.copy())
    cv2.destroyAllWindows()
    (l_cX, l_cY, l_peri, l_approx, l_ellipse) = l_cnt
    (r_cX, r_cY, r_peri, r_approx, r_ellipse) = r_cnt

    # Looks like cv2.fitEllipse returns: 
    #
    #       ((x_centre,y_centre),(minor_axis,major_axis),angle)
    #
    # The `angle` is what we want. The minor/major axes don't change with rotation.
    print("l_ellipse: {}".format(l_ellipse))
    print("r_ellipse: {}".format(r_ellipse))

    # The `angle` is measured in [0,180]. Convert to yaw in [-90,90].
    l_angle = l_ellipse[2]
    r_angle = r_ellipse[2]


if __name__ == "__main__":
    """ See the top of the file for program-wide arguments. """
    arm, _, d = utils.initializeRobots(sleep_time=2)
    arm.home()
    arm.close_gripper()
    print("arm home: {}".format(arm.get_current_cartesian_position()))

    # Test with angles. NOTE: no need to have yaw be outside [-89.9, 90].
    #utils.show_images(d)
    #rotations = [ (-90, 10, -170) for i in range(9)]
    #motion_planning(d.left_contours_by_size, d.left_image, arm, rotations)

    ## Test which rotations make sense.
    #for rot in rotations:
    #    print("we are moving to rot {}".format(rot))
    #    utilities.move(arm, HOME_POS, rot, SPEED_CLASS)
    #    time.sleep(3)

    # Test proof of concept of the need for the automatic trajectory collection.
    #proof_of_concept_part_one(d.left_image.copy(), 
    #                          d.right_image.copy(), 
    #                          d.left_contours, 
    #                          d.right_contours, 
    #                          arm)
    #proof_of_concept_part_two(d.left_image.copy(), 
    #                          d.right_image.copy(), 
    #                          d.left_contours, 
    #                          d.right_contours, 
    #                          arm)

    # Now test if we can detect seed orientations.
    detect_seed_orientation(d.left_image.copy(), 
                            d.right_image.copy(), 
                            d.left_contours, 
                            d.right_contours, 
                            arm, d)
