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
        self.left_image_proc = None
        #self.left_image_circles = None

        self.right_image = None
        self.right_image_gray = None
        self.right_image_proc = None
        #self.right_image_circles = None

        # A bunch of *lists*, i.e. contours/circles, for detecting stuff.
        self.left_contours = []
        self.left_contours_by_size = []
        #self.left_circles = []
        self.right_contours = []
        self.right_contours_by_size = []
        #self.right_circles = []

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


    def right_info_callback(self, msg):
        if self.info['r']:
            return
        self.info['r'] = msg


    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.left_image_proc = IMAGE_PREPROCESSING_DEFAULT(self.left_image)

        self.left_contours = self.get_contours(self.left_image_proc)
        self.left_contours_by_size = self.get_contours_by_size(self.left_image_proc)

        self.left_image_gray = IMAGE_PREPROCESSING_DEFAULT(self.left_image, grayscale_only=True)
        #self.left_circles = self.get_circles_list(self.left_image_gray)
        #self.left_image_circles = self.set_circles(self.left_image_gray.copy(), self.left_circles)


    def right_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        self.right_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.right_image_proc = IMAGE_PREPROCESSING_DEFAULT(self.right_image)

        self.right_contours = self.get_contours(self.right_image_proc)
        self.right_contours_by_size = self.get_contours_by_size(self.right_image_proc)

        self.right_image_gray = IMAGE_PREPROCESSING_DEFAULT(self.right_image, grayscale_only=True)
        #self.right_circles = self.get_circles_list(self.right_image_gray)
        #self.right_image_circles = self.set_circles(self.right_image_gray.copy(), self.right_circles)


    def get_contours(self, img):
        (cnts, _) = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        processed_countours = []

        for c in cnts:
            try:
                # Approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                # Find the centroids of the contours in _pixel_space_. :)
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                processed_countours.append((cX, cY, approx, peri))
            except:
                pass

        # Sort contours in rough left to right, up to down ordering.
        processed_countours = sorted(processed_countours, key = lambda x: x[0])
        processed_countours = sorted(processed_countours, key = lambda x: x[1])
        return processed_countours


    def get_contours_by_size(self, img):
        (cnts, _) = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_by_size = sorted(cnts, key=cv2.contourArea, reverse=True)
        return contours_by_size


    def get_circles_list(self, img):
        """ 
        Let's try this, maybe detecting circles for our phantom tissues is easier. 
        I might need to tweak dp and minDist, of course. Note that this takes a while to run.
        EDIT: yeah, let's forget about this.
        """
        circles = cv2.HoughCircles(image=img.copy(), method=cv2.cv.CV_HOUGH_GRADIENT, dp=1.2, minDist=10)
        return circles


    def set_circles(self, img, circles):
        """ 
        *Sets* the circles here and then returns it. Inputs are a copy of the grayscale
        image along with a list of circles for that image (from `self.get_circles_list`).
        EDIT: not using this, since the act of calling `cv2.HoughCircles` takes ages for some reason.
        """
        th = 4

        if circles is not None:
            circles_int = np.round(circles[0,:]).astype("int")
            # loop over the (x, y) coordinates and radius of the circles
            for (x,y,r) in circles_int:
                # draw circle in the appropriate image, and a rectangle corresponding to its center
                cv2.circle(img=img, center=(x,y), radius=r, color=(0,255,0), thickness=th)
                cv2.rectangle(img, (x-5,y-5), (x+5,y+5), (0,128,255), -1)
        return img
