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


psm1 = robot("PSM1")

psm1.open_gripper(90)
time.sleep(2)


d = DataCollector()

time.sleep(1)

img = cv2.medianBlur(d.left_image[:,850:], 9) #[:, 580:1500]
mask = cv2.inRange(img, np.array((100,100,100),dtype = "uint8"), np.array((255,255,255),dtype = "uint8"))
#mask = cv2.inRange(img, np.array((50,50,50),dtype = "uint8"), np.array((100,100,100),dtype = "uint8"))
output = np.sign(cv2.bitwise_and(img, img, mask = mask))*255
output =  cv2.erode(output,np.array([21,21]),iterations = 1)
#output = cv2.morphologyEx(output, cv2.MORPH_OPEN, np.array([51,51]))


import matplotlib.pyplot as plt
plt.imshow(output, cmap='gray')



estimates = np.argwhere(output[:,:,0] > 0)
tree = BallTree(estimates, leaf_size=2)
N,p = estimates.shape

i = np.random.choice(np.arange(0,N))

dist, ind = tree.query(estimates[i,:], k=50)
mean = np.mean(estimates[ind[0],:].T, axis=1)
cov = np.cov(estimates[ind[0],:].T)
U, V = np.linalg.eig(cov)
minor = np.argmin(U)

xl = 0.00603172613036
xh = 0.173861855068

yl = 0.00569858956424
yh = 0.170927597351



yr = (yh-yl)*(mean[0]/1080) + yl - 0.010*(1-(mean[1]+850)/1920)
xr = (xh-xl)*(1-(mean[1]+850)/1920) + xl


ax = plt.axes()
ax.arrow(mean[1], mean[0], 100*V[1,minor], 100*V[0,minor], head_length=30, fc='r', ec='r')
plt.show()

exit()


angle = np.arctan(V[1,minor]/V[0,minor])*180/np.pi


estimate = (xr, yr, angle)

#exit()


#import pickle 
#reg = pickle.load(open('model.p','rb'))
#out = reg.predict(np.array([xr, yr, np.sin(angle), np.cos(angle)]))
#print(out, xr, yr, angle)

pos = [xr,yr,-.111]

#print(angle)

#pos = [xr, yr, -.151]

rot = tfx.tb_angles(angle, 0.0,-160.0)

psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.005)

time.sleep(4)

#xr = xr - 0.01*np.sin(angle)
#yr = yr + 0.01*np.cos(angle)

pos = [xr,yr, -.158]

rot = tfx.tb_angles(angle,0.0,-160.0)

psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.005)

time.sleep(4)



"""
psm1.close_gripper()



pos[2] = pos[2] + 0.04
psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.005)

time.sleep(2)

pos[0] = xl - 0.02
psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.005)
time.sleep(2)

psm1.open_gripper(90)
"""





