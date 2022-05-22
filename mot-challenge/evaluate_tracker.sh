#!/usr/bin/env bash

# Evaluate trackers
python3 mot-challenge/TrackEval-master/scripts/run_mot_challenge.py \
    --USE_PARALLEL=True --NUM_PARALLEL_CORES=6 \
    --GT_FOLDER=mot-challenge/data/gt \
    --TRACKERS_FOLDER=mot-challenge/data/trackers \
    --CLASSES_TO_EVAL=pedestrian \
    --BENCHMARK=MOT16 \
    --SPLIT_TO_EVAL=train \
    --PRINT_CONFIG=True \
    --DO_PREPROC=False \
    --TRACKER_SUB_FOLDER=mot-challenge/data \
