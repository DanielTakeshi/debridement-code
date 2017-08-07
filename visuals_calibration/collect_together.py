""" 
Once we've manually run a bunch of stuff in `click_and_crop.py`, this takes that saved 
pickle file and plots things together into one plot. The data was stored like this:

data_pt = {'target_pos': pos,
           'target_rot': ROTATION,
           'actual_pos': new_pos,
           'actual_rot': new_rot,
           'center_target_pixels': (cX,cY),
           'center_actual_pixels': CENTER_OF_BOXES[-1]}

I.e. each element in the pickle file is a python dictionary.
"""

import cv2
import numpy as np
import os
import pickle
import sys

# Double check these as needed.
ORIGINAL_IMAGE  = 'images/check_calibration/calibration_blank_image.jpg'
IM_VISUALS_DIR  = 'images/visuals/'
IM_VISUALS_FILE = 'visuals_calibration/data_v00.p'


def load_data(filename):
    data = []
    f = open(filename,'rb')
    while True:
        try:
            d = pickle.load(f)
            assert len(d.keys()) == 6
            data.append( (d['center_target_pixels'], d['center_actual_pixels']) )
        except EOFError:
            break
    return data


def make_fancy_image(image, visuals):
    """ Make a fancy image, e.g. for a paper or presentation.

    Parameters
    ----------
    image:
        A clone of the original image for calibration, can draw on it.
    visuals:
        Each item consists of *two* tuples, for target and actual, respectively.
    """
    radius = 7
    im_v01 = image.copy()

    for index,val in enumerate(visuals):
        pt_targ, pt_real = val

        # Handle first image.
        cv2.circle(img=image, center=pt_targ, radius=radius, color=(255,0,0), thickness=-1)
        cv2.circle(img=image, center=pt_real, radius=radius, color=(0,0,255), thickness=-1)
        #cv2.putText(img=image, text="{}".format(pt_targ), org=pt_targ, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)
        #cv2.putText(img=image, text="{}".format(pt_real), org=pt_real, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,255), thickness=2)

        # Handle second image.
        cv2.circle(img=im_v01, center=pt_targ, radius=radius, color=(255,0,0), thickness=-1)
        cv2.circle(img=im_v01, center=pt_real, radius=radius, color=(0,0,255), thickness=-1)
        # Argh, this is not part of the interface, I think if I upgrade OpenCV I can do this.
        # cv2.arrowedLine(img=im_v01, pt1=pt_targ, pt2=pt_real, color=(0,0,0), thickness=1)
        cv2.line(img=im_v01, pt1=pt_targ, pt2=pt_real, color=(0,0,0), thickness=2)

    cv2.imshow("Image v01", im_v01)
    key = cv2.waitKey(0)
    cv2.imwrite(IM_VISUALS_DIR+'calib_image_v00.png', image)
    cv2.imwrite(IM_VISUALS_DIR+'calib_image_v01.png', im_v01)


if __name__ == "__main__":
    image_original = cv2.imread(ORIGINAL_IMAGE).copy()
    visuals_data = load_data(IM_VISUALS_FILE)
    make_fancy_image(image_original, visuals_data)

    #for i, (cX, cY, approx, peri) in enumerate(d.left_contours):  
    #    image = image_original.copy()
    #    cv2.circle(img=image, center=(cX,cY), radius=50, color=(0,0,255), thickness=1)
    #    cv2.drawContours(image=image, contours=[approx], contourIdx=-1, color=(0,255,0), thickness=3)
    #    cv2.imshow("Press ESC if this isn't a desired contour (or a duplicate), press any other key to proceed w/robot movement.", image)
    #    firstkey = cv2.waitKey(0) 

    #    if firstkey not in ESC_KEYS:
    #        # First, determine where the robot will move to based on the pixels.
    #        target = np.squeeze( arm_map.predict([[cX,cY]]) )
    #        pos = [target[0], target[1], ARM1_ZCOORD]
    #        rot = tfx.tb_angles(ROTATION[0], ROTATION[1], ROTATION[2])

    #        # Robot moves to that point and will likely be off. I think 6 seconds is enough for the camera to refresh.
    #        arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)
    #        time.sleep(6)

    #        # Get the updated image and put the center coordinate there (make it clear). Blue=Before, Red=AfteR.
    #        updated_image_copy = (d.left_image).copy()
    #        cv2.circle(img=updated_image_copy, center=(cX,cY), radius=6, color=(255,0,0), thickness=-1)
    #        cv2.putText(img=updated_image_copy, text="{}".format((cX,cY)), org=(cX,cY), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255,0,0), thickness=2)

    #        # Now we apply the callback and drag a box around the end-effector on the (updated!) image.
    #        window_name = "Robot has tried to move to ({},{}). Click and drag a box around its end effector. Then press any key (or ESC if I made mistake).".format(cX,cY)
    #        cv2.namedWindow(window_name)
    #        cv2.setMouseCallback(window_name, click_and_crop)
    #        cv2.imshow(window_name, updated_image_copy)
    #        key = cv2.waitKey(0)
    #        if key in ESC_KEYS:
    #            continue

    #        # Now save the image with the next available index. It will contain the contours.
    #        index = len(os.listdir(IMDIR))
    #        cv2.imwrite(IMDIR+"/point_"+str(index).zfill(2)+".png", updated_image_copy)
    #        cv2.destroyAllWindows()
 
    #        # Get position and orientation of the arm, save, & reset. Be careful that I understand `data_pt` ordering! 
    #        # I.e. the target_pos is what the random forest predicted for the target position.
    #        frame = arm.get_current_cartesian_position()
    #        new_pos = tuple(frame.position[:3])
    #        new_rot = tfx.tb_angles(frame.rotation)
    #        new_rot = (new_rot.yaw_deg, new_rot.pitch_deg, new_rot.roll_deg)
    #        data_pt = {'target_pos': pos,
    #                   'target_rot': ROTATION,
    #                   'actual_pos': new_pos,
    #                   'actual_rot': new_rot,
    #                   'center_target_pixels': (cX,cY),
    #                   'center_actual_pixels': CENTER_OF_BOXES[-1]}
    #        storeData(OUTPUT_FILE, data_pt)

    #        # Some stats for debugging, etc.
    #        num_added += 1
    #        print("contour {}, data_pt: {} (# added: {})".format(i, data_pt, num_added))
    #        assert (2*num_added) == (2*len(CENTER_OF_BOXES)) == len(POINTS)

    #    else:
    #        print("(not storing contour {})".format(i))

    #    arm.home()
    #    arm.close_gripper()
    #    cv2.destroyAllWindows()
