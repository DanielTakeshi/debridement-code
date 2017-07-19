import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.misc

from robot import *
from data_collector import DataCollector

from sklearn.neighbors import BallTree

import time

import Tkinter as tk

import pickle

import matplotlib.pyplot as plt

import random

import string


def logTimeStep(collector, arm, sn, filename, nextpos, nextrot, gripper, stop, focus):
    f = open('demonstrations/'+filename, 'a')
    frame = arm.get_current_cartesian_position()
    pos = tuple(frame.position[:3])
    rot = tfx.tb_angles(frame.rotation)
    rot = (rot.yaw_deg, rot.pitch_deg, rot.roll_deg)

    pickle.dump({'pos': pos, 
                 'rot': rot, 
                 'sn': sn, 
                 'image': collector.left_image.dumps(), 
                 'npos': nextpos, 
                 'nextrot': nextrot,
                 'gripper': gripper,
                 'stop': stop,
                 'focus': focus}, f)

    f.close()




#demo file name
filename = ''.join(random.choice(string.lowercase) for _ in range(9)) + '.p'


#initialize whereever it is
psm1 = robot("PSM1")
psm1.open_gripper(90)
time.sleep(2)



#do some image processing
d = DataCollector()

time.sleep(1)

img = cv2.medianBlur(d.left_image[:,850:], 7) #[:, 580:1500]
mask = cv2.inRange(img, np.array((100,100,100),dtype = "uint8"), np.array((255,255,255),dtype = "uint8"))
output = np.sign(cv2.bitwise_and(img, img, mask = mask))*255
output =  cv2.erode(output,np.array([7,7]),iterations = 1)


estimates = np.argwhere(output[:,:,0] > 0)
tree = BallTree(estimates, leaf_size=2)
N,p = estimates.shape

while True:
    i = np.random.choice(np.arange(0,N))

    dist, ind = tree.query(estimates[i,:], k=25)
    mean = np.mean(estimates[ind[0],:].T, axis=1)
    cov = np.cov(estimates[ind[0],:].T)
    U, V = np.linalg.eig(cov)
    minor = np.argmin(U)
    yangle = np.arctan(V[1,minor]/V[0,minor])*180/np.pi

    if np.abs(yangle) < 20:

        plt.imshow(output, cmap='gray')
        ax = plt.axes()
        ax.arrow(mean[1], mean[0], 100*V[1,minor], 100*V[0,minor], head_length=30, fc='r', ec='r')
        plt.show()


        action = raw_input('Play action')

        if action == 'y':
            focus = [mean[1], mean[0], yangle]
            logTimeStep(d, psm1, 0, filename, [], [], False, False, focus)
            break
        else:
            exit()


#pick based on image heuristic
yangle = 0.5*np.arctan(V[1,minor]/V[0,minor])*180/np.pi
pangle =  (yangle < 0)*(90 - yangle) / 6
rangle = -(170.0 - (90 - yangle) / 6)

regx, regy = pickle.load(open('model-sep.p','rb'))
xr = regx.predict(np.array([mean[1], mean[0]]))[0]
yr = regy.predict(np.array([mean[1], mean[0]]))[0]



#first non-trivial action
pos = [xr,yr,-.151]
rot =  tfx.tb_angles(0.0,0.0,-160.0)

logTimeStep(d, psm1, 1, filename, pos, rot, False, False, focus)

psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.005)

time.sleep(4)


pos = [xr,yr, -.162]

rot = tfx.tb_angles(yangle, 0.0,-160.0)

logTimeStep(d, psm1, 2, filename, pos, rot, False, False, focus)

psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.005)

time.sleep(4)



logTimeStep(d, psm1, 3, filename, pos, rot, True, False, focus)

psm1.close_gripper()


pos[2] = pos[2] + 0.03
logTimeStep(d, psm1, 4, filename, pos, rot, True, False, focus)
psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.005)
time.sleep(2)


action = raw_input('Play action')

if action == 'y':
    pass
else:
    logTimeStep(d, psm1, 5, filename, pos, rot, True, True, focus)
    psm1.open_gripper(90)

    action = raw_input('Ready?')

    if action == 'y':
        psm1.close_gripper()
        time.sleep(1)

        logTimeStep(d, psm1, 4, filename, pos, rot, True, False, focus)
        psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.005)
        time.sleep(2)

    else:
        exit()




action = raw_input('Play action')

if action == 'y':
    pos[0] = -0.02
    pos[2] = pos[2] + 0.02

    logTimeStep(d, psm1, 5, filename, pos, rot, True, False, focus)

    psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.005)
    time.sleep(2)

    logTimeStep(d, psm1, 6, filename, pos, rot, False, True, focus)

    psm1.open_gripper(90)


else:
    pos = (0.126742903205, 0.0619060066152, -0.129995445247)

    logTimeStep(d, psm1, 5, filename, pos, rot, True, False, focus)

    psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.005)
    time.sleep(2)

    logTimeStep(d, psm1, 6, filename, pos, rot, False, True, focus)

    psm1.open_gripper(90)


time.sleep(1)

pos = ( 0.055,  0.035, -0.111)

rot = tfx.tb_angles(0.0,0.0,-160.0)
    
psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)
    
time.sleep(1)





