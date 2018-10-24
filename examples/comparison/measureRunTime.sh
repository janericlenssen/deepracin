#!/bin/bash

#pyScript="featuresFFT.py"
#pyScript="featuresWVT.py"
pyScript="build_dr_model.py"

counter=1
sum=0
while [ $counter -le 1000 ]
do
    term=$(python ${pyScript} | grep -oP '(?<=Whole graph took: )[\d.]+')
    echo "$term" | bc -l
    sum=$(bc -l <<< "$sum + $term")
    ((counter++))
done
echo "****DONE****"
echo "Average:"
echo "$sum / 1000" | bc -l
