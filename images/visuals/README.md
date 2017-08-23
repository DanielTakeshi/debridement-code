# Stats

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
mean:   13.3650867628
median: 13.3199376596
max:    26.6833281283
min:    1.0
std:    7.84835511714
```


## Version 03

With the better random forest, BUT with an extra 0.0005 decrease in the x-coordinate!

```
Distances among the pixels, (x,y) only:
mean:   10.0342690983
median: 7.76316057026
max:    23.769728648
min:    1.0
std:    6.60657074399
```

## Version 04

With the better random forest, BUT with an extra 0.001 decrease in the x-coordinate!

```
Distances among the pixels, (x,y) only:
mean:   7.4614609319
median: 6.32455532034
max:    17.691806013
min:    0.0
std:    4.56665945078
```

Awesome!


## Version 05

With a 1.5mm decrease in the x-coordinate.

```
Distances among the pixels, (x,y) only:
mean:   12.9640766292
median: 12.1448007947
max:    24.3515913238
min:    1.0
std:    6.18146381764
```
