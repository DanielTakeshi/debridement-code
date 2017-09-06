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



# Gripper, with Auto Trajectories

Thus, 20 through 24 have yaw roughly 90, 45, 0, -45, -90.

## Version 20

Yaw is roughly +90

```
Distances among the pixels, (x,y) only:
mean:   22.5858126052
median: 25.0
max:    53.310411741
min:    2.2360679775
std:    13.8644843432
```

## Version 21

Yaw is roughly +45

```
Distances among the pixels, (x,y) only:
mean:   21.5303981882
median: 21.0
max:    39.1152144312
min:    7.07106781187
std:    7.48420887129
```

## Version 22

Yaw is roughly 0

```
Distances among the pixels, (x,y) only:
mean:   22.7911384968
median: 22.6715680975
max:    39.4461658466
min:    11.401754251
std:    5.93474084341
```

## Version 23

Yaw is roughly -45

```
Distances among the pixels, (x,y) only:
mean:   28.3757370169
median: 29.4108823397
max:    38.9486841883
min:    18.3575597507
std:    4.79170177419
```

(This one seems a bit weird, has errors that are more uniform. Well, I guess human guidance should help ...)


## Version 24

Yaw is roughly -90

```
Distances among the pixels, (x,y) only:
mean:   26.7276890758
median: 22.0227155455
max:    47.8852795753
min:    12.7279220614
std:    9.10897874673
```

## Version 25

Yaw is randomly chosen uniformly in [-90, 90].


```
Distances among the pixels, (x,y) only:
mean:   23.1320045825
median: 21.0237960416
max:    48.8773976394
min:    3.60555127546
std:    9.7201745131
```


## Version 30

This is version 20 but with a RF predictor on top of that. The results are much better!

```
Distances among the pixels, (x,y) only:
mean:   8.95279720253
median: 8.48528137424
max:    28.0
min:    1.0
std:    6.19829418646
```

## Version 31

This is version 21 but with a RF predictor on top of that. The results are better! They could have been better if the pixel locations agreed among the two cameras (I think that was the reason for two of the outliers...).

```
Distances among the pixels, (x,y) only:
mean:   12.1006710636
median: 11.6619037897
max:    30.0
min:    2.0
std:    7.10951393423
```

## Version 32

This is version 22 but with a RF predictor on top of that. The results are better!

```
Distances among the pixels, (x,y) only:
mean:   12.8575833722
median: 12.0
max:    27.2029410175
min:    1.41421356237
std:    7.15918838964
```

## Version 33

This is version 23 but with a RF predictor on top of that. The results are better!

Note that version 23 seemed to be very bad so this is worse but it's still about a 2x improvement in terms of 2x reduced error.

```
Distances among the pixels, (x,y) only:
mean:   14.2161071535
median: 14.1421356237
max:    29.1547594742
min:    1.0
std:    6.41333523449
```

## Version 34

This is version 24 but with a RF predictor on top of that. The results are better!

```
Distances among the pixels, (x,y) only:
mean:   10.5123774498
median: 8.0622577483
max:    25.0599281723
min:    1.0
std:    5.94533962111
```

## Version 35

This is version 25 but with a RF predictor on top of that. The results are better!

Results are worse than the version 33 which was the worst but not that much and the median was better ... we kind of expect the general case to be slightly worse due to rounding, etc.

```
Distances among the pixels, (x,y) only:
mean:   14.7649220238
median: 13.0
max:    34.9284983931
min:    2.2360679775
std:    8.53379452876
```


# Rigid Body Only

40 thorugh 44 are for yaw (roughly) 90, 45, 0, -45, -90.

## Version 40

I used:

```
python scripts/click_and_crop_v3.py --use_rigid_body --version_in 1 --version_out 40 --fixed_yaw 90 --z_offset 0.004
```

I've observed that height is a main reason for low performance.

```
Distances among the pixels, (x,y) only:
mean:   44.1015196016
median: 43.0813184571
max:    67.2681202354
min:    29.1547594742
std:    8.11692923813
```

## Version 41

```
python scripts/click_and_crop_v3.py --use_rigid_body --version_in 1 --version_out 41 --fixed_yaw 45 --z_offset 0.006
```
Wow, height was really a problem.

```
Distances among the pixels, (x,y) only:
mean:   72.0347893859
median: 66.4078308635
max:    167.242339137
min:    32.0156211872
std:    31.3462775802
```

## Version 42

```
python scripts/click_and_crop_v3.py --use_rigid_body --version_in 1 --version_out 42 --fixed_yaw 0 --z_offset 0.008
```

```
Distances among the pixels, (x,y) only:
mean:   51.9589669664
median: 53.712196008
max:    85.7088093489
min:    29.0
std:    15.1216980458
```

## Version 43

```
python scripts/click_and_crop_v3.py --use_rigid_body --version_in 1 --version_out 43 --fixed_yaw -45 --z_offset 0.009
```

I wised up and put in minimum and maximum commands for the z-coordinate. For safety.

```
Distances among the pixels, (x,y) only:
mean:   53.616406975
median: 50.7740090991
max:    78.1024967591
min:    43.2781700168
std:    8.82015809427
```

## Version 44

```
python scripts/click_and_crop_v3.py --use_rigid_body --version_in 1 --version_out 44 --fixed_yaw -90 --z_offset 0.008
```

```
Distances among the pixels, (x,y) only:
mean:   39.8973511551
median: 36.7695526217
max:    59.6657355607
min:    26.4007575649
std:    10.5302936849
```

## Version 45

```
python scripts/click_and_crop_v3.py --use_rigid_body --version_in 1 --version_out 45 --z_offset 0.008
```

```
Distances among the pixels, (x,y) only:
mean:   47.1539969817
median: 48.3735464898
max:    75.133215025
min:    19.7230829233
std:    12.2597132369
```
