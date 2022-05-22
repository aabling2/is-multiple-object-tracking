#!/usr/bin/env bash

# Generate results from tracker
python3 -m mot-challenge.mot_challenge \
    --sequence_dir=datasets/MOT16/test/MOT16-06 \
    --detection_file=datasets/MOT16/resources/detections/MOT16_POI_test/MOT16-06.npy \
    --min_confidence=0.8 \
    --nn_budget=5 \
    --display=True \
    --mot_dir=datasets/MOT16 \
    --detection_dir=datasets/MOT16/resources/detections/MOT16_test \
    --output_dir=mot-challenge/data/results
