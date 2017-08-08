# Visuals

This will attempt to make visuals for calibration tests.

- `data_v00.p`, 36 points, what I showed in Ken's lab meeting on August 7.
- `data_v01.p`, 35 points (I missed one by mistake), same as version 0 except I arranged contours in reverse order.
- `data_v02.p`, unfortunately this one ended early because I think I pressed the space bar twice, see the image it's clear where it went wrong. This, by the way, was a repeat of version 0 so it went through the contours from the top left to bottom right. I was trying to see if I could perfectly replicate it. And it seems like it does that to some degree but I will investigate.

# Stats

## Version 00

```
Distances among the pixels, (x,y) only:
mean:   34.096251805
median: 33.5975147119
max:    77.4144689318
min:    7.21110255093
std:    15.1107412703
```

## Version 01

```
Distances among the pixels, (x,y) only:
mean:   25.9132747254
median: 24.1660919472
max:    66.7532770731
min:    3.16227766017
std:    15.4259306319
```

## Version 02

(Remember not to put too much stock into these results given the outliers that I reported earlier.)

```
Distances among the pixels, (x,y) only:
mean:   31.0256805753
median: 23.3773174406
max:    154.728794993
min:    8.0622577483
std:    29.3284357721
```
