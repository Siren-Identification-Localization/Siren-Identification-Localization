#!/bin/bash
for i in `seq 1 89`;
do
    python shift_wav.py -i beep/440Hz.wav -o beep/440Hz_${i}shift.wav -n ${i}
done    
