#!/bin/bash

export DISPLAY=""
# DEFINE GLOBAL PARAMS
jobs=3
table="/BIRL/io/inputs/dataset_ANHIR/dataset_medium.csv"
# this folder has to contain bland of images and landmarks
dataset="/BIRL/io/inputs/dataset_ANHIR/images"
results="/BIRL/io/outputs/dataset_ANHIR/"
apps="/BIRL/Applications"

preprocessings=("gray")

for pproc in "${preprocessings[@]}"
do

    python bm_experiments/bm_elastix.py \
         -t $table \
         -d $dataset \
         -o $results \
         -elastix "$apps/elastix/bin" \
         -cfg ./configs/elastix_rigid.txt \
         -pproc $pproc \
         --visual --unique --nb_workers $jobs

done
