# Calibration


- `calib_circlegrid_left_v00_ONELIST.p` and the one for `right` contain ONE LIST with all the data needed. They are of length 36.

- The `backup` directory has the original pickle files where we have to load one by one. I had to make the lists since I mistakenly double counted one of the circles (but I knew which one, don't worry...) and manually removed it. The point is, it's easier to load it in with one list, so I changed it to do that!

- Use `mapping_results` to save reslts from `mapping.py`, a VERY important script which does the full camera pixel to robot frame pipeline.

- Use `calibration_results` to store stuff from `click_and_crop.py` for visualizing calibration errors in practice. UPDATE: `data_vXY.p` will be for that click and crop, but `data_for_rf_vXY.p` is going to be where I put the human data from `train_rf.py` Don't get confused!.








# Older Stuff (don't look at it...)

Do not look at this ..

I used `final_calib` version 0, but I messed up the cameras. I'm now using version 1 and this time I saved the bounding box so I can cross-reference with this later. Hopefully that will resolve things.
I'm also trying to do my visuals for the random forest regressor to see where calibration is going wrong.

EDIT: also added a linear regression map. Thus use one of the two:

- `daniel_final_mono_map_01.p` (random forest with 100 trees)
- `daniel_final_mono_lin-regression_01.p` (normal linear regression with intercept term)

I tried normalizing the data for linear regression but found no improvement. I saved the tuple of meanX and stdX in `mean_X,std_X.p`, though I am unlikely to need it.

# Even Older Stuff

The stuff below are in the `old_configs` and thus not to be used.

## Version 01

This was my first attempt, done Tuesday July 11. It's a bit rough, I wouldn't use this as I got results that were quite off when I tried. The dataset is fairly small.

## Version 02

Second attempt, Friday July 14. Gah, the ICRA deadline is in *two months* and I have nothing but calibratoin done. I HAVE TO GET MOVING!!

- I used a fixed setup where I put in two hard drives in place to prevent the tissue phantom from moving too much. 
- I decided I would allow *two* calibrations per circle, instead of one as I did earlier.
- I ignored the entire right column as that one is at too different a z-coordinate.
- I closed the grippers, not sure if this matters but it's easier for me if there is one sharp point.
- I am allowed to "press down" on the paper so that it makes contact with the tissue surface, but then I don't force it after that.


## Version 03

Third attempt, same as the second one except I rotated the phantom tissue by 180 degrees. Combining this with v02 brings the random forest traning error down to almost 1e-6.
