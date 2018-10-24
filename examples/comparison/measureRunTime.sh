#!/bin/bash

pyScript="featuresFFT.py"
#pyScript="featuresWVT.py"
#pyScript="build_dr_model.py"

counter=1
sum=0
while [ $counter -le 10 ]
do
    sum=$(bc -l <<< "$sum + $(python ${pyScript} | grep -oP '(?<=Whole graph took: )[\d.]+')")
    ((counter++))
done
echo "$sum / 10" | bc -l
