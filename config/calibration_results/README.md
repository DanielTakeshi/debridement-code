# Calibration Results

## Random Forest Files

This is for making that BETTER random forest predictor.

- Gripper: `data_for_rf_v00.p`
- Scissors: `data_for_rf_v10.p`

## Data Files

Gripper:

- Version 00: from `click_and_crop.py` rigid body only. Unfortunately I must have added a few things by mistake since there are 41 of them, so I made `data_v00_36only.p` with ideally the correct 36.
- Version 01: from `click_and_crop.py` rigid body plus the bad random forest.
- Version 02: the data file for the rigid body + better random forest regressor, from `click_and_crop_v2.py`, no x offset.

Scissors:

- Version 10: rigid body only.
- Version 11: rigid body + bad random forest.
- Version 12: rigid body + good random forest.
