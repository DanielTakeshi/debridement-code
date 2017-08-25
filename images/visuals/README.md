# Stats

First gripper, followed by scissors.

# First Gripper

## Version 00

(Skipping since there are 41 data points, I _think_ I can fix by deleting the first five.)

UPDATE: I think I fixed it. Results:

```
Distances among the pixels, (x,y) only:
mean:   30.4723009163
median: 29.5125340648
max:    42.1900462195
min:    23.7065391823
std:    4.46840627556
```

## Version 01

(Skipping since it's with the bad RF and also incomplete.)

Never mind, we should have it.

```
Distances among the pixels, (x,y) only:
mean:   31.0500129492
median: 31.3128541783
max:    41.1096095822
min:    23.0217288664
std:    3.82710018844
```

The reason why it's bad is because the rigid body "thinks" it is already so good, so there is very little room for improvement.

## Version 02

With the better random forest.

```
Distances among the pixels, (x,y) only:
mean:   5.03540693236
median: 4.472135955
max:    12.6491106407
min:    1.0
std:    3.14116208556
```

Niiiice.




# Scissors

## Version 10

Wow, a rigid body by itself is good! The RF might only be useful for the last three rows of the grid.

```
Distances among the pixels, (x,y) only:
mean:   7.64911082659
median: 5.89414452228
max:    18.973665961
min:    1.0
std:    5.79674172908
```

## Version 11

Bad RF, actually worse though it's probably in the noise. Reason why its hard for this to work is that the robot thinks the rigid body is already great.

```
Distances among the pixels, (x,y) only:
mean:   9.03511991963
median: 8.0622577483
max:    19.1049731745
min:    1.0
std:    5.27888321881
```

## Version 12

Good RF. Niiiice! The only thing it had to do was increase the y coordinate by a millimeter for the last two rows.

```
Distances among the pixels, (x,y) only:
mean:   4.17630719943
median: 4.62132034356
max:    12.7279220614
min:    1.0
std:    2.51166442344
```
