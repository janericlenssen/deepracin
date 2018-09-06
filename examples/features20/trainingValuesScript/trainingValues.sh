#!/bin/bash
# Gets the training data by calling getFeatures.py on every training image
# train_dir="training/train/neg"

train_dir="training/train/neg"
declare -i iteration=1
cd ..
for entry in "${train_dir}"/*
do
    imageID="${entry:19:-4}"
    echo "iteration: ${iteration}, image: ${imageID}"
    python getFeatures.py --imageName=${entry:19} --id=${imageID}
    iteration=$((iteration+1))
    #break
done
