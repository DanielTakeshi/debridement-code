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

        self.right_image = None
        self.proc_right_image = None
        self.left_image = None
        self.proc_left_image = None
        self.right_contours = []
        self.left_contours = []

        self.info = {'l': None, 'r': None}
        self.bridge = cv_bridge.CvBridge()
        self.timestep = 0

        self.identifier = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))


        rospy.Subscriber(camera_left_topic + camera_im_str, Image,
                         self.left_image_callback, queue_size=1)
        rospy.Subscriber(camera_right_topic + camera_im_str, Image,
                         self.right_image_callback, queue_size=1)
        rospy.Subscriber(camera_left_topic + camera_info_str,
                         CameraInfo, self.left_info_callback)
        rospy.Subscriber(camera_right_topic + camera_info_str,
                         CameraInfo, self.right_info_callback)


    def left_info_callback(self, msg):
        if self.info['l']:
            return
        self.info['l'] = msg

    def right_info_callback(self, msg):
        if self.info['r']:
            return
        self.info['r'] = msg

    def right_image_callback(self, msg):
        if rospy.is_shutdown():
            return

        self.right_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.proc_right_image = IMAGE_PREPROCESSING_DEFAULT(self.right_image)
        self.right_contours = self.get_contours(self.proc_right_image)

    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return

        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        self.proc_left_image = IMAGE_PREPROCESSING_DEFAULT(self.left_image)
        self.left_contours = self.get_contours(self.proc_left_image)



    def get_contours(self, img):
        (cnts, _) = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

        processed_countours = []

        for c in cnts:
            try:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                processed_countours.append((cX, cY, approx, peri))

            except:
                pass

        processed_countours = sorted(processed_countours, key = lambda x: x[0])
        processed_countours = sorted(processed_countours, key = lambda x: x[1])

        return processed_countours