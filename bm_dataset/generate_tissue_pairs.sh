#!/bin/bash

echo Tissue?
read tissue

echo Percentage?
read percentage

echo Extension?
read extension

for f in /io/inputs/dataset_ANHIR/images/${tissue}_*; do
    if [ -d "$f" ]; then
        # Will not run if no directories are available
        python /io/BIRL/bm_dataset/generate_regist_pairs.py \
            -i "$f/scale-${percentage}pc/*.$extension" \
            -l "/io/inputs/dataset_ANHIR/landmarks/$(basename $f)/scale-${percentage}pc/*.csv" \
            -csv "/io/inputs/dataset_ANHIR/${tissue}_dataset_medium_${percentage}pc.csv" \
            --mode each2all
    fi
done