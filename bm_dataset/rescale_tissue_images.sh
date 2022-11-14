#!/bin/bash

echo Tissue?
read tissue

echo Initial percentage?
read init_percentage

echo Initial extension?
read init_extension

echo Wanted percentage?
read percentage

echo Wanted extension?
read extension

for f in /io/inputs/dataset_ANHIR/images/${tissue}_*; do
    if [ -d "$f" ]; then
        # Will not run if no directories are available
        python /io/BIRL/bm_dataset/rescale_tissue_images.py \
            -i "$f/scale-${init_percentage}pc/*.$init_extension" \
            --scales $percentage \
            -ext ".$extension" \
            --nb_workers 1
    fi
done