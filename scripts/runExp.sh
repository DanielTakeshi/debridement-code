#!/bin/bash

#make a temporary file for the experimental log
tfile=$(mktemp /tmp/exp.XXXXXXXXX)

#execute the main in the background and print the pid makes it easier to kill if something goes wrong
#but that messes up the i/o
python main.py $* >> $tfile 2>&1 & PROC_ID=$!

echo "PID of running experiment  $PROC_ID"

#spin until the program is done
while kill -0 "$PROC_ID" >/dev/null 2>&1; do
    echo $(date) 'Still working...log in:' $(wc -l $tfile)
    sleep 1
done

git add config/* results/*

git commit -m main.py $*


