#!/usr/bin/env python3
import cv2
import numpy as np
from reconstruction.epipolar_geometry import PointsReconstructor


if __name__ == "__main__":

    # Carrega imagens
    # img1 = cv2.imread('/home/augusto/Documents/mot-pis-ros/images/epipolar/left.jpg', 0)
    # img2 = cv2.imread('/home/augusto/Documents/mot-pis-ros/images/epipolar/right.jpg', 0)
    img1 = cv2.imread('/home/augusto/Documents/mot-pis-ros/datasets/EPFL-RLC/multiclass_ground_truth_images/c3/00000001.jpg', 0)
    img2 = cv2.imread('/home/augusto/Documents/mot-pis-ros/datasets/EPFL-RLC/multiclass_ground_truth_images/c4/00000001.jpg', 0)

    # Calibra parâmetros de reconstrução
    reconstructor = PointsReconstructor()
    reconstructor.calibrate(frames=[img1, img2])

    # Teste
    # reconstructor.reconstruct(points=np.array([[[0, 0]], [[10, 10]]]))
    reconstructor.reconstruct(points=reconstructor.keypoints)
    reconstructor.render_scene()

    # Match keypoints
    reconstructor.match_keypoints_homography(frames=[img1, img2])
