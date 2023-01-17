#!/bin/bash

# export FIRST_STEP="only"
# DEFINE GLOBAL PARAMS

name="bm_elastix"
jobs=3

echo Tissue?
read tissue_input

echo Percentage?
read percentage

if [ $tissue_input == "all" ] && [ $percentage == "high" ];
then
    tissue=("breast" "COAD" "gastric" "kidney" "lung-lesion" "lung-lobes" "mice-kidney")
    tissue2=("20" "25" "15" "25" "100" "100" "25")
elif [ $tissue_input == "all" ] && [ $percentage == "low" ];
then
    tissue=("breast" "COAD" "gastric" "kidney" "lung-lesion" "lung-lobes" "mice-kidney")
    tissue2=("2.5" "10" "2.5" "2.5" "2.5" "10" "5")
else
    tissue=("$tissue_input")
    tissue2=("$percentage")
fi

for i in "${!tissue[@]}"
do
    ti=${tissue[i]}
    perc=${tissue2[i]}
    for f in /io/inputs/dataset_ANHIR/images/${ti}_*; do
        if [ -d "$f" ]; then
            # Will not run if no directories are available
            python /io/BIRL/bm_dataset/generate_regist_pairs.py \
                -i "$f/scale-${perc}pc/*" \
                -l "/io/inputs/dataset_ANHIR/landmarks/$(basename $f)/scale-${perc}pc/*.csv" \
                -csv "/io/inputs/dataset_ANHIR/bm_elastix/${ti}_dataset_medium_${perc}pc.csv" \
                -rot "/io/inputs/dataset_ANHIR/initial_rotation.csv" \
                --mode each2all
        fi
    done
done