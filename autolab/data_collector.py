import numpy as np
import rospy
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
import cv2
import cv_bridge
import numpy as np
import scipy.misc
import pickle
import imutils
import time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from config.constants import *
import string
import random

"""
The DataCollector class polls data from the rostopics periodically. It manages 
the messages that come from ros.
"""
class DataCollector:

    def __init__(self, 
                 camera_left_topic="/endoscope/left/",
                 camera_right_topic="/endoscope/right/",
                 camera_info_str='camera_info',
                 camera_im_str='image_rect_color'):

        # A bunch of different *images* depending on pre-processing.
        self.left_image = None
        self.left_image_gray = None
        self.left_image_bbox = None
        self.left_image_proc = None
        self.left_image_circles = None

        self.right_image = None
        self.right_image_gray = None
        self.right_image_bbox = None
        self.right_image_proc = None
        self.right_image_circles = None

        # To get a bounding box of points, to filter away any nonsense.
        # For the initial open loop policy for the tissues that Steve gave me.
        self.lx, self.ly, self.lw, self.lh = 625, 40, 850, 820        
        self.rx, self.ry, self.rw, self.rh = 550, 40, 850, 820
        self.left_apply_bbox  = True
        self.right_apply_bbox = True

        # A bunch of *lists*, i.e. contours/circles, for detecting stuff.
        self.left_contours = []
        self.left_contours_by_size = []
        self.left_circles = []
        self.right_contours = []
        self.right_contours_by_size = []
        self.right_circles = []

        # Random stuff now, plus the subscribers!!
        self.timestep = 0
        self.info = {'l': None, 'r': None}
        self.bridge = cv_bridge.CvBridge()
        self.identifier = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
        rospy.Subscriber(camera_left_topic + camera_im_str, Image, self.left_image_callback, queue_size=1)
        rospy.Subscriber(camera_right_topic + camera_im_str, Image, self.right_image_callback, queue_size=1)
        rospy.Subscriber(camera_left_topic + camera_info_str, CameraInfo, self.left_info_callback)
        rospy.Subscriber(camera_right_topic + camera_info_str, CameraInfo, self.right_info_callback)


    def left_info_callback(self, msg):
        if self.info['l']:
            return
        self.info['l'] = msg
        #pickle.dump(self.info['l'], open('config/camera_info_matrices/left.p', 'w'))


    def right_info_callback(self, msg):
        if self.info['r']:
            return
        self.info['r'] = msg
        #pickle.dump(self.info['r'], open('config/camera_info_matrices/right.p', 'w'))


    def left_image_callback(self, msg):
        """ 
        Manages the left camera's image feed plus associated modifications and data. 
        Yeah, the bounding box (if we use it) is really quite manual.
        """
        if rospy.is_shutdown():
            return
        x,y,w,h = self.lx, self.ly, self.lw, self.lh

        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.left_image_proc = IMAGE_PREPROCESSING_DEFAULT(self.left_image)
        self.left_image_gray = IMAGE_PREPROCESSING_DEFAULT(self.left_image, grayscale_only=True)
        self.left_image_bbox = self.make_bounding_box(self.left_image_gray.copy(), x,y,w,h)

        self.left_contours = self.get_contours(self.left_image_proc, self.left_apply_bbox)
        self.left_contours_by_size = self.get_contours_by_size(self.left_image_proc, 
                self.left_apply_bbox, x,y,w,h)

        # Handle circles ... must be handled delicately. Apply bbox if desired.
        self.left_circles = self.get_circles_list(self.left_image_proc, self.left_apply_bbox, x,y,w,h)
        self.left_image_circles = self.set_circles(self.left_image_gray.copy(), self.left_circles)


    def right_image_callback(self, msg):
        """ Similar as the left camera. """
        if rospy.is_shutdown():
            return
        x,y,w,h = self.rx, self.ry, self.rw, self.rh

        self.right_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.right_image_proc = IMAGE_PREPROCESSING_DEFAULT(self.right_image)
        self.right_image_gray = IMAGE_PREPROCESSING_DEFAULT(self.right_image, grayscale_only=True)
        self.right_image_bbox = self.make_bounding_box(self.right_image_gray.copy(), x,y,w,h)

        self.right_contours = self.get_contours(self.right_image_proc, self.right_apply_bbox)
        self.right_contours_by_size = self.get_contours_by_size(self.right_image_proc, 
                self.right_apply_bbox, x,y,w,h)

        # Handle circles ... must be handled delicately. Apply bbox if desired.
        self.right_circles = self.get_circles_list(self.right_image_proc, self.right_apply_bbox, x,y,w,h)
        self.right_image_circles = self.set_circles(self.right_image_gray.copy(), self.right_circles)


    def make_bounding_box(self, img, x,y,w,h):
        """ Make a bounding box. """
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        return img


    def get_circles_list(self, img, apply_bbox, xx,yy,ww,hh):
        """ 
        Let's try this, maybe detecting circles for our phantom tissues is easier. 
        I might need to tweak dp and minDist, of course. Note that this takes a while to run.
        Also, this max radius here is just a heuristic. Otherwise we'd get a LOT of large circles.
        The method is really quite sensitive to the parameter settings. :-(
        Note: circles is a numpy array with shape (1,T,3) where T = number of circles. I am returning
        something of shape (T',3) where T' <= T since it filters out those outside the bounding box,
        assuming that `apply_bbox == True` of course.
        """
        circles = cv2.HoughCircles(image=img.copy(), method=cv2.cv.CV_HOUGH_GRADIENT, 
                dp=4.0, minDist=20.0, maxRadius=30)
        if (apply_bbox) and (circles is not None):
            real_circles = []
            for (x,y,r) in circles[0,:]:
                if (xx < x < xx+ww) and (yy < y < yy+hh):
                    real_circles.append([x,y,r])
            return real_circles
        else:
            return circles


    def set_circles(self, img, circles):
        """ 
        *Sets* the circles here and then returns it. Inputs are a copy of the grayscale
        image along with a list of circles for that image (from `self.get_circles_list`).
        Note: I'm calling this with circles of shape (K,3) where K is the number of circles.
        """
        th = 3
        rr = 3

        if circles is not None:
            circles_int = np.round(circles).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x,y,r) in circles_int:
                # draw circle in the appropriate image, and a rectangle corresponding to its center
                cv2.circle(img=img, center=(x,y), radius=r, color=(0,255,0), thickness=th)
                cv2.rectangle(img, (x-rr,y-rr), (x+rr,y+rr), (0,128,255), -1)
        return img


    def get_contours(self, img, apply_bbox):
        """ TODO: Enforce the apply_bbox condition. """
        (cnts, _) = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        processed_countours = []

        for c in cnts:
            duplicates = []
            try:
                # Approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                # Find the centroids of the contours in _pixel_space_. :)
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if self._not_duplicate(duplicates, cX, cY, rtol=5):
                    duplicates.append((cX,cY))
                    processed_countours.append((cX, cY, approx, peri))
            except:
                pass

        # Sort contours in rough left to right, up to down ordering.
        processed_countours = sorted(processed_countours, key = lambda x: x[0])
        processed_countours = sorted(processed_countours, key = lambda x: x[1])
        ## # Reverse direction!
        ## processed_countours = sorted(processed_countours, key = lambda x: x[0], reverse=True)
        ## processed_countours = sorted(processed_countours, key = lambda x: x[1], reverse=True)
        return processed_countours


    def get_contours_by_size(self, img, apply_bbox, xx,yy,ww,hh):
        """ 
        Returns the contours in order of size, largest to smallest.  Also applies the bounding 
        box condition if desired. Ideally, I'd like to enforce a convexity condition, but it's 
        hard because even the "circle" contours it finds aren't convex sets. In addition, I
        apply a rough form of duplicate contour detection if we're using bounding boxes.
        """
        (cnts, _) = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contained_cnts = []  # New list w/bounding box condition

        if apply_bbox:
            duplicates = []
            for c in cnts:
                try:
                    # Find the centroids of the contours in _pixel_space_. :)
                    M = cv2.moments(c)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # Enforce it to be within bounding box AND away from each other by a certain `rtol` pixels.
                    if (xx < cX < xx+ww) and (yy < cY < yy+hh) and self._not_duplicate(duplicates, cX, cY, rtol=100):
                        contained_cnts.append(c)
                        duplicates.append((cX,cY))
                except:
                    pass
        else:
            contained_cnts = cnts # Just keep the old list

        contours_by_size = sorted(contained_cnts, key=cv2.contourArea, reverse=True)
        return contours_by_size


    def _not_duplicate(self, duplicates, cX, cY, rtol):
        """ Helper method for checking duplicates. """
        if len(duplicates) == 0:
            return True
        for x in range(-rtol,rtol+1):
            for y in range(-rtol,rtol+1):
                if (x+cX, y+cY) in duplicates:
                    # We're close enough that it's a duplicate.
                    return False
        return True


    def get_left_bounds(self):
        return self.lx, self.ly, self.lw, self.lh


    def get_right_bounds(self):
        return self.rx, self.ry, self.rw, self.rh
