#!/bin/bash -x
FLIP=$(($(($RANDOM%10))%2))
if [ $FLIP -eq 1 ]
then
    python scripts/open_loop_seeds_auto.py --version_in 1 --version_out 28 --max_num_add 8 --close_angle 30 --z_offset -0.002 --zoffset_safety 0.006
else
    python scripts/open_loop_seeds_auto.py --version_in 1 --version_out 29 --max_num_add 8 --close_angle 30 --z_offset -0.003 --zoffset_safety 0.006 --no_rf_correctors
fi
