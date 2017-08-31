# Proofs of Concepts

## Case 1, Same Camera Points

```
(seita-py2-env)davinci0@davinci0:~/danielseita/icra2018$ python scripts/rotations_sandbox.py
arm home: [(-0.000,  0.059, -0.130),(yaw:0.5, pitch:1.2, roll:-159.8)]
(left, right) = ([896, 661], [826, 666])
(cx,cy,cz) = (-0.0194, 0.0054, 0.1887)

pos,rot -90 yaw:
(0.0499, 0.0661, -0.1632), (-88.0708, 8.7017, -159.0875)

pos,rot after manual +90 yaw change:
(0.0527, 0.0647, -0.1638), (92.7972, -32.1017, 162.1418)

pos,rot +90 yaw:
(0.0525, 0.0639, -0.1642), (89.4014, -11.0428, -176.7663)
```

I kept a seed in the same spot. The first position listed, with yaw=-90 (actually -88.0708...), is for a yaw=-90 position which can grip the seed. What happens if were to just rotate that? We would be off by a lot. See my image. You cannot just rotate to yaw=90 and expect things to be the same. (This is not printed in the output here, BTW, it is NOT the "pos,rot after manual +90 yaw change", sorry for the confusion. I just have it on camera)

Thus, the last position listed is for a yaw=90 case which CAN grip the seed. But it requires a much different position. Thus our function f must be such that:

```
f(cx, cy, cz, -88.0708, 8.7017, -159.0875)  = (0.0499, 0.0661, -0.1632)
f(cx, cy, cz, 89.4014, -11.0428, -176.7663) = (0.0525, 0.0639, -0.1642)
```

and these are NOT equal, despite having the same camera points! But the point is, for the SAME CAMERA POSITION, SAME SEED, these are the angles that work. Thus, given these camera points, plus a target yaw, we infer desired pitch and roll. Then all that is needed is to call this function to get the actual robot positions that will work.

These are subtle points.

- If you **just** rotate a gripper, the position of the robot in `x,y,z` SHOULD differ a bit since rotation will in some way impact the end-effector location.
- But if you want to pick up a seed, some further additional adjustment is needed in pitch, roll and additional movement (due to "rotation mechanisms..." for lack of a better description).

Thus there are really two extra sources of uncertainty in the robot position.


## Case 2, Different Camera Points

I was going to do this, but it is kind of redundant. Basically, if we are given a seed and a gripper that at yaw=-90 can pick it up, if we just change yaw to +90, the robot position does not actually change. **UPDATE: actually I don't think this is right ... please ignore this discussion** But the seed must be shifted to allow the robot to grip it at the current gripper. Thus the camera points are different, even if the robot targets are the same, i.e.:

```
f(cx, cy, cz, -90, pitch, roll) = f(cx', cy', cz', +90, pitch, roll)
```

But isn't this implied by the first case? The first case clearly shows the motivation for automatic trajectory collection of data.
