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


def collect(event, arm1, estimate):
    """ 
    Here, data.p is a pickle file with a bunch of pos, rot, and estimate stuff. Keep loading
    the file and you'll see lots of points.  But why do we never use `event`?
    """
    f = open('data.p', 'a')

    # Daniel: THIS is how we make use of the cartesian position. I get it.
    frame = arm1.get_current_cartesian_position()
    pos = tuple(frame.position[:3])
    rot = tfx.tb_angles(frame.rotation)
    rot = (rot.yaw_deg, rot.pitch_deg, rot.roll_deg)

    pickle.dump({'pos': pos, 'rot': rot, 'estimate': estimate}, f)
    f.close()
    exit()


# Start robot, open gripper, blah blah blah.
psm1 = robot("PSM1")
psm1.open_gripper(90)
time.sleep(2)
estimate = None

# Use Tkinter, make root window, bind shift-down so that we keep calling collect when pressing the shift key.
# See http://effbot.org/tkinterbook/tkinter-events-and-bindings.htm for documentation details.
root = tk.Tk()
root.geometry('300x200')
text = tk.Text(root, background='black', foreground='white', font=('Comic Sans MS', 12))
text.pack() # just packs the window to fit the text...
root.bind('<Shift-Down>', lambda event, arm1=psm1: collect(event, arm1, estimate) )

# First get amask with cv2.inRange, then "and" it with the original image, to get a black/white binary image.
d = DataCollector()
time.sleep(1)
img = cv2.medianBlur(d.left_image[:,850:], 13) #[:, 580:1500]
mask = cv2.inRange(img, np.array((100,100,100),dtype = "uint8"), np.array((255,255,255),dtype = "uint8"))
#mask = cv2.inRange(img, np.array((50,50,50),dtype = "uint8"), np.array((100,100,100),dtype = "uint8"))
output = np.sign(cv2.bitwise_and(img, img, mask = mask))*255

# Not sure what's going on here?
estimates = np.argwhere(output[:,:,0] > 0)
tree = BallTree(estimates, leaf_size=2)
N,p = estimates.shape
i = np.random.choice(np.arange(0,N))
dist, ind = tree.query(estimates[i,:], k=50)
mean = np.mean(estimates[ind[0],:].T, axis=1)
cov = np.cov(estimates[ind[0],:].T)
U, V = np.linalg.eig(cov)
minor = np.argmin(U)

# Some hard-coded values, I assume because we are dealing with a level z-plane?
xl = 0.00603172613036
xh = 0.173861855068
yl = 0.00369858956424
yh = 0.105927597351
yr = (yh-yl)*(mean[0]/1080) + yl
xr = (xh-xl)*(1-(mean[1]+850)/1920) + xl

# Don't know what is going on here.
angle = np.arctan(V[1,minor]/V[0,minor])*180/np.pi
estimate = (xr, yr, angle)
xr = xr
pos = ( xr,  yr, -0.118189139536)
rot = tfx.tb_angles(angle,0,-160)
psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.005)
pos = ( xr,  yr, -0.158189139536)
psm1.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.005)
time.sleep(2)

# Not sure why we're plotting here?
import matplotlib.pyplot as plt
plt.imshow(output, cmap='gray')
ax = plt.axes()
ax.arrow(mean[1], mean[0], 100*V[1,minor], 100*V[0,minor], head_length=30, fc='r', ec='r')
plt.show()

# Why are we just now showing the main loop? I don't get it.
root.mainloop()
