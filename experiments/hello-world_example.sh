#!/bin/bash

export HELLOWORLD="smthg"
jobs=3

dataset="/io/inputs/dataset_ANHIR/images"
table="/io/inputs/dataset_ANHIR/hello-world_dataset_medium.csv"
apps="/BIRL/Applications"

python experiments/low-high_res_elastix.py \
    -t $table \
    -d $dataset \
    -o "/outputs" \
    -elastix "$apps/elastix/bin" \
    -cfg "./configs/elastix_LowRes_affine.txt" \
    -sgm_params "/io/inputs/dataset_ANHIR/segmentation_ps/ps_06.json" \
    --visual --unique --nb_workers $jobs
