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


Gripper (auto collection):

- Versions 20 through 25: see `images/visuals/README.md` for details.
- Files named `data_human_guided_vXY.p` are those corresponding to the same experimental setting as `data_vXY.p` except that they have human guidance... I have one for each of the data versions now (as of September 1, 2017).

UPDATE: I now have a set of 40 through 45, lol ..., e.g. `data_human_guided_v40.p` et al were created with these commands:

```
python scripts/human_guidance_auto.py --version_in 1 --version_out 40 --fixed_yaw 90
python scripts/human_guidance_auto.py --version_in 1 --version_out 41 --fixed_yaw 45
python scripts/human_guidance_auto.py --version_in 1 --version_out 42 --fixed_yaw 0
python scripts/human_guidance_auto.py --version_in 1 --version_out 43 --fixed_yaw -45
python scripts/human_guidance_auto.py --version_in 1 --version_out 44 --fixed_yaw -90
python scripts/human_guidance_auto.py --version_in 1 --version_out 45
```

Note that `--version_in 1` will automatically load in the `auto_params_matrices_v01.p` file which contains the model directory for the deep neural network.

(Edit: OK I'm not doing 45 since we don't use it anyway ... I really have to get moving to real experiments, anyway!)
