# Automatic Trajectory Collector

I *really* hope this automatic trajectory collector will work.

Run `scripts/automatic_trajectory_collector.py` to get data. Run it twice. The first helps to determine the safe boundaries for the stuff I can command. The second is when I actually run things.

You can rename a `guidelines.p` file to a backup in case you want to regenerate one of them.

This is the output I got after running the trajectory collector to first get the "guidelines":

```
key,val = min_x,0.00398412300361
key,val = min_y,0.01534833957
key,val = max_pitch,50
key,val = max_roll,-100
key,val = min_pitch,-50
key,val = z_gamma,-0.164024948938
key,val = z_alpha,-0.12825066709
key,val = min_yaw,-90
key,val = min_roll,-180
key,val = z_beta,0.00606683643819
key,val = max_yaw,90
key,val = max_x,0.0657019637084
key,val = max_y,0.076835347471
P:
[[ 0.00398412  0.06993424 -0.16428698]
 [ 0.06030757  0.07683535 -0.17112625]
 [ 0.06570196  0.01930596 -0.1725109 ]
 [ 0.01209506  0.01534834 -0.16529797]]
A:
[[ 0.00811592  0.00636646  0.14208872]
 [ 0.00636646  0.01140276  0.18142389]
 [ 0.14208872  0.18142389  4.        ]]
x:
[-0.12825067  0.00606684 -0.16402495]
b:
[-0.02430834 -0.03050537 -0.6732221 ]
```
