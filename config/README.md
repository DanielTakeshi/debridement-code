# Calibration

I used `final_calib` version 0, but I messed up the cameras. I'm now using version 1 and this time I saved the bounding box so I can cross-reference with this later. Hopefully that will resolve things.
I'm also trying to do my visuals for the random forest regressor to see where calibration is going wrong.


(Note: some of them are in the `old_configs` if I'm no longer using them.)

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
