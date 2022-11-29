#!/bin/bash

export DISPLAY=""
# DEFINE GLOBAL PARAMS
jobs=2

echo Tissue?
read tissue

if [[ $tissue == "all" ]];
then
    low_tissue=("coad" "gastric" "kidney" "lung-lesion" "lung-lobes" "mammary-gland" "mice-kidney")
    low_tissue2=("10" "2.5" "2.5" "2.5" "10" "10" "5")
else
    low_tissue=("$tissue")

    echo Percentage?
    read percentage
    low_tissue2=("$percentage")
fi 

echo Config?
read config

if [[ $config == "all" ]];
then
    configs=("xavier" "birl" "LowRes")
else
    configs=($config)
fi

# this folder has to contain bland of images and landmarks
dataset="/io/inputs/dataset_ANHIR/images"
parameter_sets=/io/inputs/dataset_ANHIR/segmentation_ps/*.json
apps="/BIRL/Applications"

for i in "${!low_tissue[@]}"
do
    ti=${low_tissue[i]}
    perc=${low_tissue2[i]}
    table="/io/inputs/dataset_ANHIR/${ti}_dataset_medium_${perc}pc.csv"   
    results="/io/outputs/dataset_ANHIR/PS_evaluation/${ti}" 
    
    for ps in $parameter_sets
    do
        ps_name=$(basename -- "$ps" .json)
        mkdir ${results}/${ps_name}

        for c in "${configs[@]}"
        do

            python experiments/low-high_res_elastix.py \
                -t $table \
                -d $dataset \
                -o ${results}/${ps_name} \
                -elastix "$apps/elastix/bin" \
                -cfg "./configs/elastix_${c}_affine.txt" \
                -sgm_params $ps \
                --visual --unique --nb_workers $jobs

        done
    done
done