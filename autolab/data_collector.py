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
import os
import random
import string

import json

from robot import *

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped

class DataCollector:
    """
    The DataCollector class polls data from the rostopics periodically
    """

    def __init__(self, 
                 camera_left_topic="/endoscope/left/",
                 camera_right_topic="/endoscope/right/",
                 camera_info_str='camera_info',
                 camera_im_str='image_rect_color'):

        self.right_image = None
        self.left_image = None
        self.info = {'l': None, 'r': None}
        self.bridge = cv_bridge.CvBridge()
        self.timestep = 0

        self.directory = ''.join(random.choice(string.lowercase) for _ in range(9))
        os.mkdir(self.directory)


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

    def left_image_callback(self, msg):
        if rospy.is_shutdown():
            return
        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")


    def resetDirectory(self):
        self.directory = ''.join(random.choice(string.lowercase) for _ in range(9))
        os.mkdir(self.directory)
        self.timestep = 0


    def log(self, fn, params):
        f = open(self.directory+"/"+'kin.txt', 'a')
        f.write(json.dumps({fn: params}))
        f.close()

        scipy.misc.imsave(self.directory+'/left'+str(self.timestep)+'.png', self.left_image)
        scipy.misc.imsave(self.directory+'/right'+str(self.timestep)+'.png', self.right_image)

        self.timestep = self.timestep + 1