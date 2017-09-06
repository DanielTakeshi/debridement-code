# Images

For images after my UAI trip. Note that some of the images that were used to check calibration are not here.

- `left_image.jpg` used for drawing stuff on.

## Check regressors

- Version 00 is for the rigid body only mapping.
- Version 01 is for the rigid body AND random forest mapping (but it's not the correct way to use the RF corrector...).
- Version 02 uses the rigid body and the BETTER random forest mapping, and no x-offsets.

- Version 10 is for the scissors with the rigid body only mapping.
- Version 11, scissors, RB + internal RF
- Version 12, scissors, RB + human-guided RF.

Versions 2X are for mappings from the neural networks.

- Version 20 for the first net I tried ...

## Visuals

(There's a README in this directory which contains important statistics!!)

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


**For neural net stuff**:

Using DNN + RF

- `seeds_v20`: pumpkin seeds, random orientation. I made an error with numbering, 
    trials 1 and 2 correspond to indices 0 and 1, but then for some reason index 
    2 seems to not correspond to a trial. So the others after that are trial X for index X.

- `seeds_v21`: sunflower seeds, random orientation. This time the numbering is right, 
    i.e. trial X corresponds to index X-1.

Note that the 15 trials with v20 and v21 used the slower timing benchmark ... ugh.

Using DNN ONLY as a baselie:

- `seeds_v22`: pumpkin seeds, random orientation.

- `seeds_v23`: sunflower seeds, random orientation.

This time v22 and v23 have good timing benchmarks, even if their performance sucks.

- `seeds_v24` debugging pumpkin seeds ... for some reason my results are worse? Darn ... we just have to argue that this task is very challenging!

- `seeds_v25` raisins, DNN+RF

- `seeds_v26` raisins, DNN only
