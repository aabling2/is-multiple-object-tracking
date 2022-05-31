#!/usr/bin/env bash

# Generate results from tracker
python3 -m mot-challenge.run_tracker \
    --sequence_dir=datasets/MOT16/test/MOT16-06 \
    --detection_file=datasets/MOT16/resources/detections/MOT16_POI_test/MOT16-06.npy \
    --detection_dir=datasets/MOT16/resources/detections/MOT16_test \
    --mot_dir=datasets/MOT16