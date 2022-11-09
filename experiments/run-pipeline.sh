#!/bin/bash

export DISPLAY=""
# DEFINE GLOBAL PARAMS
jobs=3

echo Tissue?
read tissue

echo Percentage
read percentage

echo Config?
read config


if [[ $config == "all" ]];
then
    configs=("xavier" "birl" "LowRes")
else
    configs=($config)
fi

table="/io/inputs/dataset_ANHIR/${tissue}_dataset_medium_${percentage}pc.csv"
# this folder has to contain bland of images and landmarks
dataset="/io/inputs/dataset_ANHIR/images"
results="/io/outputs/dataset_ANHIR/"
apps="/BIRL/Applications"

for c in "${configs[@]}"
do

    python experiments/low-high_res_elastix.py \
         -t $table \
         -d $dataset \
         -o $results \
         -elastix "$apps/elastix/bin" \
         -cfg "./configs/elastix_${c}_affine.txt" \
         --visual --unique --nb_workers $jobs

done
