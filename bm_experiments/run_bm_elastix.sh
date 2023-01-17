#!/bin/bash

name="bm_elastix"
jobs=1

echo Tissue?
read tissue_input

if [[ $tissue_input == "all" ]];
then
    tissue=("coad" "gastric" "kidney" "lung-lesion" "lung-lobes" "mice-kidney")
    tissue2=("25" "15" "25" "100" "100" "25")
else
    tissue=("$tissue_input")

    echo Percentage?
    read percentage
    
    tissue2=("$percentage")
fi 

# this folder has to contain bland of images and landmarks
dataset="/io/inputs/dataset_ANHIR/images"
apps="/BIRL/Applications"

for i in "${!tissue[@]}"
do
    ti=${tissue[i]}
    perc=${tissue2[i]}
    table="/io/inputs/dataset_ANHIR/bm_elastix/${ti}_dataset_medium_${perc}pc.csv"   
    results="/io/outputs/dataset_ANHIR/PS_evaluation/${ti}" 
    
    mkdir ${results}/${name}
    
    python bm_experiments/bm_elastix.py \
        -t $table \
        -d $dataset \
        -o ${results}/${name} \
        -elastix "$apps/elastix/bin" \
        -cfg "./configs/elastix_bspline.txt" \
        -pproc "gray" \
        # --visual \
        --unique --nb_workers $jobs

done