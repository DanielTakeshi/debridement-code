#!/bin/bash -x
FLIP=$(($(($RANDOM%10))%2))
if [ $FLIP -eq 1 ]
then
    python scripts/open_loop_seeds_auto.py --version_in 1 --version_out 30 --max_num_add 8 --close_angle 30 --z_offset -0.002 --zoffset_safety 0.010
else
    python scripts/open_loop_seeds_auto.py --version_in 1 --version_out 31 --max_num_add 8 --close_angle 30 --z_offset -0.002 --zoffset_safety 0.010 --no_rf_correctors
fi
