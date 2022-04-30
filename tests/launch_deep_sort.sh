#!/usr/bin/env bash

# Generating detections
# python3 tools/generate_detections.py \
#     --model=resources/networks/mars-small128.pb \
#     --mot_dir=../../datesets/MOT16/train \
#     --output_dir=./resources/detections/MOT16_train

# Running the tracker
python3 ../deep_sort/deep_sort_app.py \
    --sequence_dir=../datasets/MOT16/test/MOT16-06 \
    --detection_file=../datasets/MOT16/resources/detections/MOT16_POI_test/MOT16-06.npy \
    --min_confidence=0.3 \
    --nn_budget=100 \
    --display=True