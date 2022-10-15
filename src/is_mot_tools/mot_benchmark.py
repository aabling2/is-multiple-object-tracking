#!/usr/bin/env python3
from DeepSORT.deep_sort import nn_matching
from DeepSORT.deep_sort.detection import Detection
from DeepSORT.deep_sort.tracker import Tracker as deep_sort


trackers = [
    deep_sort(metric=nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance=0.2, nn_budget=None)),
]


def main():
    print("okok")


if __name__ == '__main__':
    main()
