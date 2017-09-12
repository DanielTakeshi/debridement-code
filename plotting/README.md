This is not actually the GitHub but a quick script I'm using to make a plot for
the paper.

The directory `keras_results_old` has the old results for meters, bleh.

Use the data in `data`. The training data is NOT NOT NOT normalized, but I have
the mean and standard deviation there ... also, the x,y,z stuff is in
millimeters and the rotations are in angles (in degrees, i.e. "Euler
angles"...).


- `results_kfolds10_v00.npy` uses 500 epochs for each neural net (but John did
  5000).
- `results_kfolds10_v01.npy` uses 5000 epochs for each neural net, to make it
  more in line with John's performance. TODO overnight...
