# Scripts

- **Running the Code**: use `open_loop_seeds.py` to get all the data. It automatically saves the original images so I can verify that the failure cases were due to perception and not calibration.


- **Training Rigid Body + Random Forest**: Very important: `mapping.py` develops mappings between pixels to robot/camera stuff. 

  - Then, `click_and_crop.py` will test if the stuff in `mapping.py` worked, basically the manual thing where I check if it goes to the right spot, and `collect_together.py` will put it in one image for me.

  - Also, `train_rf.py` (and then `train_rf_make_rf.py`) should ideally make the full pipeline work by getting the data (then training the RF) on my actual data where I correct based on my own data. Then `click_and_crop_v2.py` will test that with the different (but better, in my opinion) method for dealing with random forest residuals.


- **Initial Calibration Data Gathering**: `calibrate_onearm.py` is when I do my manual stuff with the DVRK's arms, then:

  - `check_calib_points.py` is purely debugging

  - `check_camera_location.py` is for checking the original camera position, MAYBE useful if the camera gets hit and its location changes. But in general there's no easy way to recover from that so BE CAREFUL! I better remind people.


- **Miscellaneous**: 

  - Use `measure_speed.py` for measuring speed. I got:

  ```
  times (in seconds!!) for speed class Slow:
  [ 5.63329911  5.97371697  5.51606894  5.81516409  5.46755385  5.66470695
    5.70830917  5.59941387  5.88957787  5.9694159 ]
  mean(times): 5.72
  std(times):  0.17

  times (in seconds!!) for speed class Medium:
  [ 3.85610509  3.83264089  3.85077     3.70209599  3.78492498  3.68619084
    3.92790604  3.68224502  3.89478898  3.84750009]
  mean(times): 3.81
  std(times):  0.08

  times (in seconds!!) for speed class Fast:
  [ 1.50204897  1.46190405  1.49993706  1.46128392  1.47281504  1.53321815
    1.50067091  1.458601    1.51267982  1.46655107]
  mean(times): 1.49
  std(times):  0.02
  ```

  This is for a 10cm translation. Thus, the speeds I should report are:

  ```
  Slow:   1.75 cm/sec
  Medium: 2.62 cm/sec
  Fast:   6.71 cm/sec
  ```

  - Use `test_color_detection.py` to test if I can detect colors in an image. Unfortunately this is unlikely to work.
