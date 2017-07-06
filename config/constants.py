"""
This file describes a number of useful constants and offsets for the DVRK
"""
import cv2

#PSM1 home position
HOME_POSITION_PSM1 = ((0.00, 0.06, -0.13), (0.0, 0.0,-160.0))

#PSM2 home position
HOME_POSITION_PSM2 =  ((-0.015,  0.06, -0.10), (180,-20.0, 160))

#Safe speed
SAFE_SPEED = 0.005

#Fast speed
FAST_SPEED = 0.03

#set the parameters of the default image processing
def IMAGE_PREPROCESSING_DEFAULT(img):
    img = cv2.medianBlur(img, 9)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, 7, 13, 13)
    return cv2.Canny(img,100,200)
