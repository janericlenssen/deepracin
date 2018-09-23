#!/bin/bash
# Gets the training data by calling getFeatures.py on every training image

#19, 24
#train_dir="training/train/pos"
#train_dir="training/testposneg/neg"
train_dir="training/testposneg/pos"
#train_dir = "training/test"

declare -i iteration=1
cd ..
for entry in "${train_dir}"/*
do
    imageID="${entry:24:-4}"
    echo "iteration: ${iteration}, image: ${imageID}"
    python getFeatures.py --imageName=${entry:24} --id=${imageID}
    iteration=$((iteration+1))
    #break
done
