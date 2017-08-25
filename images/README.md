# Images

For images after my UAI trip. Note that some of the images that were used to check calibration are not here.

- `left_image.jpg` used for drawing stuff on.

## Check regressors

- Version 00 is for the rigid body only mapping.
- Version 01 is for the rigid body AND random forest mapping (but it's not the correct way to use the RF corrector...).
- Version 02 uses the rigid body and the BETTER random forest mapping, and no x-offsets.
- Version 03 uses the rigid body and the BETTER random forest mapping AND x-offset of -0.5 millimeter.
- Version 04 uses the rigid body and the BETTER random forest mapping AND x-offset of -1 millimeter.
- Version 05 uses the rigid body and the BETTER random forest mapping AND x-offset of -1.5 millimeter since I didn't want the one with the best raw value to be an "extreme" point.

Version 3 might be best to use in practice since there's a slight advantage to overshooting x due to the angle of the gripper and the seed shapes.

- Version 10 is for the scissors with the rigid body only mapping.
- Version 11, scissors, RB + internal RF
- Version 12, scissors, RB + human-guided RF.

## Visuals

(There's a README in this directory.)

## Seeds

These are for saving the initial images from the open loop stuff, since I want to argue that any errors we have will be because of perception problems, not calibration.

- Version 00: for sunflower seeds, horizontal orientation, with 8 seeds (I had to fiddle a lot due to using the raw tissue phantom as the background. Unfortunately I don't think I'm using this for the paper, but at least it's good to get this data.

**Pumpkin seeds**, with 8 seeds, white background, etc.

- `seeds_v01`: slow speed
- `seeds_v03`: medium speed
- `seeds_v05`: fast speed

**Sunflower seeds**, with 8 seeds, white background, etc.

- `seeds_v02`: slow speed
- `seeds_v04`: medium speed
- `seeds_v06`: fast speed

**Debugging**:

- Version 99: for debugging, e.g. with super fast movement. :-)
