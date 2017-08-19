# Scripts

- **Training Rigid Body + Random Forest**: Very important: `mapping.py` develops mappings between pixels to robot/camera stuff. 

  - Then, `click_and_crop.py` will test if the stuff in `mapping.py` worked, basically the manual thing where I check if it goes to the right spot, and `collect_together.py` will put it in one image for me.

  - Also, `train_rf.py` (and then `train_rf_make_rf.py`) should ideally make the full pipeline work by getting the data (then training the RF) on my actual data where I correct based on my own data.

- **Initial Calibration Data Gathering**: `calibrate_onearm.py` is when I do my manual stuff with the DVRK's arms, then:

  - `check_calib_points.py` is purely debugging

  - `check_camera_location.py` is for checking the original camera position, MAYBE useful if the camera gets hit and its location changes. But in general there's no easy way to recover from that so BE CAREFUL! I better remind people.

