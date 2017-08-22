# Images

For images after my UAI trip. Note that some of the images that were used to check calibration are not here.

- `left_image.jpg` used for drawing stuff on.

## Check regressors

- Version 00 is for the rigid body only mapping.
- Version 01 is for the rigid body AND random forest mapping (but it's not the correct way to use the RF corrector...).
- Version 02 uses the rigid body and the BETTER random forest mapping, and no x-offsets.
- Version 03 uses the rigid body and the BETTER random forest mapping AND x-offset of -0.5 millimeter.
- Version 04 uses the rigid body and the BETTER random forest mapping AND x-offset of -1 millimeter.

Version 3 might be best to use in practice since there's a slight advantage to overshooting x due to the angle of the gripper and the seed shapes.

## Visuals

(There's a README in this directory.)

## Seeds

These are for saving the initial images from the open loop stuff, since I want to argue that any errors we have will be because of perception problems, not calibration.

- Version 00: for sunflower seeds, horizontal orientation, with 8 seeds (I had to fiddle a lot due to using the raw tissue phantom as the background. Unfortunately I don't think I'm using this for the paper, but at least it's good to get this data.
- Version 01: for pumpkin seeds, horizontal orientation, with 8 seeds. Fortunately, due to the white background, it's easy to detect.
- Version 02: for sunflower seeds, horizontal orientation, with 8 seeds, BUT a while background (since that makes detection easier).
- Version 03: same as version 2 (i.e. *sunflower*) except the speed is now 0.06. :-)
- Version 04: same as version 1 (i.e. *pumpkin*) except the speed is now 0.06. :-)

- Version 99: for debugging, e.g. with super fast movement. :-)
