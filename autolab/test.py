from robot import *

from data_collector import DataCollector
import cv2
import numpy as np
import pickle


def coarseGo(arm, cx, cy):
    xl = 0.00603172613036
    xh = 0.173861855068

    yl = 0.00569858956424
    yh = 0.140927597351

    yr = (yh-yl)*(cy/1080.0) + yl 
    xr = (xh-xl)*(1-(cx+850)/1920.0) + xl

    angle = np.random.choice(np.arange(-80,80,10))

    pos = [xr,yr, -.158]

    rot = tfx.tb_angles(0, 0.0,-160.0)
    
    arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.005)

    time.sleep(4)

    return angle

def servoGo(arm, cx, cy):
    regx, regy = pickle.load(open('model-sep.p','rb'))
    angle = np.random.choice(np.arange(-80,80,10))
    outx = regx.predict(np.array([cx, cy]))
    outy = regy.predict(np.array([cx, cy]))
    
    pos = [outx[0], outy[0], -0.158]

    rot = tfx.tb_angles(0.0, 0.0,-160.0)
    
    arm.move_cartesian_frame_linear_interpolation(tfx.pose(pos, rot), 0.03)

    time.sleep(4)
    


def collect(arm1, estimate):
    f = open('data.p', 'a')

    frame = arm1.get_current_cartesian_position()
    pos = tuple(frame.position[:3])
    rot = tfx.tb_angles(frame.rotation)
    rot = (rot.yaw_deg, rot.pitch_deg, rot.roll_deg)

    pickle.dump({'pos': pos, 'rot': rot, 'estimate': estimate}, f)

    f.close()



psm1 = robot("PSM1")

psm1.open_gripper(80)

time.sleep(2)



d = DataCollector()

time.sleep(2)


img = cv2.medianBlur(d.left_image[:,850:], 13)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.bilateralFilter(img, 11, 17, 17)
output = cv2.Canny(img,100,200)

(cnts, _) = cv2.findContours(output.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    print(cX, cY)

    #angle = coarseGo(psm1, cX, cY)
    servoGo(psm1, cX, cY)

    img = d.left_image[:,850:]
    cv2.circle(img, (cX, cY), 50, (0,0,255))
    cv2.drawContours(img , [approx], -1, (0, 255, 0), 3)
    cv2.imshow("Calibration", img)
    cv2.waitKey(0)

    collect(psm1, (cX,cY))



