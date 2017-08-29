# Calibration


## Automatic Calibration Data 

(From the automatic trajectory collection code.)

These go in the `grid_auto` folder.

I'll just be using one version, 00, since I can't imagine running this many times, and if I want fewer data points for ablation studies, I should do just that!

The data format should be the SAME as the manual stuff, i.e. there should be one list in the pickle file with the stuff, except it's not 36 points but MANY MORE points. And it will have already been cleaned up. :-)

- Use `keras_results` to store the networks from the `mapping.py` with the automatic setting on.


## Manual, Human Calibration Data 

**Update August 25, 2017**: I am putting the initial, human-guided calibration data in the `grid` subfolder since different rotations mean different calibrations.

- `calib_circlegrid_left_v00_ONELIST.p` and the one for `right` contain ONE LIST with all the data needed. They are of length 36.
    
    - Update: I also have version 10 now, which is for that cutting tool. The "Monopolar Curved Scissors." I have a picture of the original tool that I've been using. Oh, and by the way, version 10 has 35 points, not 36, because the right camera is utter awful.

- Use `mapping_results` to save reslts from `mapping.py`, a VERY important script which does the full camera pixel to robot frame pipeline.

- Use `calibration_results` to store stuff from `click_and_crop.py` for visualizing calibration errors in practice. UPDATE: `data_vXY.p` will be for that click and crop, but `data_for_rf_vXY.p` is going to be where I put the human data from `train_rf.py` Don't get confused!.

- The `backup` directory has the original pickle files where we have to load one by one. I had to make the lists since I mistakenly double counted one of the circles (but I knew which one, don't worry...) and manually removed it. The point is, it's easier to load it in with one list, so I changed it to do that! **UPDATE: deprecated, no longer exists, combined with `grid`.**
