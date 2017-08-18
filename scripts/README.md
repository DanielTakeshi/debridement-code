# Scripts

- Very important: `mapping.py` develops mappings between pixels to robot/camera stuff. Then, `click_and_crop.py` will test if the stuff in `mapping.py` worked, basically the manual thing where I check if it goes to the right spot.

- Calibrartion: `calibrate_onearm.py` is when I do my manual stuff with the DVRK's arms, `check_calib_points.py` is purely debugging, and `check_camera_location.py` is for checking the original camera position, MAYBE useful if the camera gets hit and its location changes. But in general there's no easy way to recover from that so BE CAREFUL! I better remind people.
