# Results from mapping

## Parameter Dictionary

Gripper

- Version 00 is the first one, with the initial 36 for each the left and right cameras. Nothing much else to say. I also print the text output, for debugging and sanity checks. Things look good here, FYI.
- Version 01 has the 90 degree yaw.

Scissors

- Version 10 is for the "Monopolar Curved Scissors." With 35 points.

## Random Forest Predictor

We're also going to put the random forest mapping here, this one trained on my human data (not done beforehand) ... version 0 is again the first trial with the 36 data points.


# Automatic Stuff

Trained from the automatically generated trajectories.

- Use anything with `auto` in its name for the neural net. 
- For rf correctors, use `rf_human_guided` stuff. Do NOT use `random_forest_predictor`. OH, if those don't have version numbers, then it was v0 ... yeah.
- OH, version 01 for the `auto` stuff now has a bunch of rigid body transformations embedded inside it. :-) One per each of the rotation discretizations.
