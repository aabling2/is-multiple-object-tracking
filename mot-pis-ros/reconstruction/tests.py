#!/usr/bin/env python3
import cv2
import argparse
import numpy as np
from math import cos, sin, pi
import matplotlib.pyplot as plt
from reconstruction.epipolar_geometry import triangulate, drawlines
from simulation.camera_projection import CameraProjection, SceneObject, ProjectionPlots


# Transformações geométricas
class Transformations():
    def __init__(self):
        self.eye = np.eye(4)

    def z_rotation(self, angle):
        angle = angle * pi/180
        rotation_matrix = np.array([
            [cos(angle), -sin(angle), 0, 0],
            [sin(angle), cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        return rotation_matrix

    def x_rotation(self, angle):
        angle = angle * pi/180
        rotation_matrix = np.array([
            [1, 0, 0, 0],
            [0, cos(angle), -sin(angle), 0],
            [0, sin(angle), cos(angle), 0],
            [0, 0, 0, 1]])
        return rotation_matrix

    def y_rotation(self, angle):
        angle = angle * pi/180
        rotation_matrix = np.array([
            [cos(angle), 0, sin(angle), 0],
            [0, 1, 0, 0],
            [-sin(angle), 0, cos(angle), 0],
            [0, 0, 0, 1]])
        return rotation_matrix

    def translation(self, dx, dy, dz):
        translation_matrix = np.array([
            [1, 0, 0, dx],
            [0, 1, 0, dy],
            [0, 0, 1, dz],
            [0, 0, 0, 1]])
        return translation_matrix

    def apply(self, obj, move_coord=[0., 0., 0.], angles=[0., 0., 0.]):
        dx, dy, dz = move_coord
        angx, angy, angz = angles

        T = self.translation(dx, dy, dz)

        if angx != 0:
            R = self.x_rotation(angx)
            M = np.dot(R, T)
            obj = np.dot(R, obj)
        elif angy != 0:
            R = self.y_rotation(angy)
            M = np.dot(R, T)
            obj = np.dot(M, obj)
        elif angz != 0:
            R = self.z_rotation(angz)
            M = np.dot(R, T)
            obj = np.dot(M, obj)
        elif angx == 0 and angy == 0 and angz == 0:
            obj = np.dot(T, obj)

        return obj


def triangulation(kp1, kp2, T_1w, T_2w):
    """Triangulation to get 3D points
    Args:
        kp1 (Nx2): keypoint in view 1 (normalized)
        kp2 (Nx2): keypoints in view 2 (normalized)
        T_1w (4x4): pose of view 1 w.r.t  i.e. T_1w (from w to 1)
        T_2w (4x4): pose of view 2 w.r.t world, i.e. T_2w (from w to 2)
    Returns:
        X (3xN): 3D coordinates of the keypoints w.r.t world coordinate
        X1 (3xN): 3D coordinates of the keypoints w.r.t view1 coordinate
        X2 (3xN): 3D coordinates of the keypoints w.r.t view2 coordinate
    """
    kp1_3D = np.ones((3, kp1.shape[0]))
    kp2_3D = np.ones((3, kp2.shape[0]))
    kp1_3D[0], kp1_3D[1] = kp1[:, 0].copy(), kp1[:, 1].copy()
    kp2_3D[0], kp2_3D[1] = kp2[:, 0].copy(), kp2[:, 1].copy()
    X = cv2.triangulatePoints(T_1w[:3], T_2w[:3], kp1_3D[:2], kp2_3D[:2])
    X /= X[3]
    X1 = T_1w[:3] @ X
    X2 = T_2w[:3] @ X
    return X[:3], X1, X2


# turn [[x,y]] -> [[x,y,1]]
def add_ones_1D(x):
    #return np.concatenate([x,np.array([1.0])], axis=0)
    return np.array([x[0], x[1], 1])
    #return np.append(x, 1)


# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    if len(x.shape) == 1:
        return add_ones_1D(x)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


# turn [[x,y,w]]= Kinv*[u,v,1] into [[x/w,y/w,1]]
def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]


def triangulate_normalized_points(pose_1w, pose_2w, kpn_1, kpn_2):
    # P1w = np.dot(K1,  M1w) # K1*[R1w, t1w]
    # P2w = np.dot(K2,  M2w) # K2*[R2w, t2w]
    # since we are working with normalized coordinates x_hat = Kinv*x, one has
    P1w = pose_1w[:3, :]  # [R1w, t1w]
    P2w = pose_2w[:3, :]  # [R2w, t2w]

    point_4d_hom = cv2.triangulatePoints(P1w, P2w, kpn_1.T, kpn_2.T)
    good_pts_mask = np.where(point_4d_hom[3] != 0)[0]
    point_4d = point_4d_hom / point_4d_hom[3]

    # return point_4d.T
    points_3d = point_4d[:3, :].T
    return points_3d, good_pts_mask


if __name__ == "__main__":

    # Pega argumentos de entrada
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--object', type=str, default=None, help='Caminho do objeto STL.')
    args = parser.parse_args()

    # ENCONTRA MATRIZES PARA TRANSFORMAÇÕES
    img1 = cv2.imread('left.jpg', 0)         # queryingImage # left image
    img2 = cv2.imread('right.jpg', 0)        # trainImage # right image

    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    pts1, pts2 = pts1[:8, :], pts2[:8, :]

    # CAMERA MATRIX
    fx, fy = (1.0, 1.0)
    sx, sy = img1.shape[:2][::-1]  # Escala da projeção por eixo
    cx, cy = (sx//2, sy//2)  # Centro óptico no plano de projeção
    cam_pos = np.array([0., 0., 0.])
    cam_rot = np.array([0., 0., 0.])
    M = Transformations()  # Transformações geométricas

    # Matriz de parâmetros intrínsecos
    Kf = np.array([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])
    Ks = np.array([[sx, 0, cx], [0, sy, cy], [0, 0, 1]])
    K = np.dot(Ks, Kf)

    # Matriz K inversa
    Kinv = np.array([[1/fx, 0, -cx/fx], [0, 1/fy, -cy/fy], [0, 0, 1]])

    # Matriz de projeção canônica
    Pi = np.append(np.eye(3), np.zeros((3, 1)), axis=1)

    # Matriz de parâmetros extrínsecos
    R = np.dot(np.dot(M.z_rotation(cam_rot[2]), M.y_rotation(cam_rot[1])), M.x_rotation(cam_rot[0]))
    t = M.translation(cam_pos[0], cam_pos[1], cam_pos[2])
    Rt = np.dot(R, t)

    # Matriz calibrada de transformação dos pontos
    P = np.dot(K, np.dot(Pi, Rt))
    # camera_matrix = P

    camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    print("Camera matrix:\n", camera_matrix)

    # RECONSTRUCTION
    # using the essential matrix can get you the rotation/translation bet. cameras,
    # although there are two possible rotations:
    E, m2 = cv2.findEssentialMat(pts1, pts2, camera_matrix, cv2.RANSAC, 0.999, 1.0)
    # Re1, Re2, t_E = cv2.decomposeEssentialMat(E)
    print("Essential matrix:\n", E)

    # recoverPose gets you an unambiguous R and t. One of the R's above does agree with the R determined here.
    # RecoverPose can already triangulate, I check by hand below to compare results.
    K_l = camera_matrix
    K_r = camera_matrix
    pts1_norm = normalize(Kinv=Kinv, pts=pts1)
    pts2_norm = normalize(Kinv=Kinv, pts=pts2)
    retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(E, pts1_norm, pts2_norm, camera_matrix, distanceThresh=0.5)
    print("Recovered pose:\n", R, t)

    # given R,t you can  explicitly find 3d locations using projection
    M_r = np.concatenate((R, t), axis=1)
    M_l = np.concatenate((np.eye(3, 3), np.zeros((3, 1))), axis=1)
    proj_r = np.dot(camera_matrix, M_r)
    proj_l = np.dot(camera_matrix, M_l)
    # proj_pts1 = np.expand_dims(pts1, axis=1)
    # proj_pts2 = np.expand_dims(pts2, axis=1)
    points, mask = triangulate_normalized_points(pose_1w=proj_l, pose_2w=proj_r, kpn_1=pts1_norm, kpn_2=pts2_norm)
    print("Triangulate points", points)
    # points_4d_hom = cv2.triangulatePoints(proj_l, proj_r, proj_pts1, proj_pts2)
    # print(points_4d_hom)
    # points_4d = points_4d_hom / np.tile(points_4d_hom[-1, :], (4, 1))
    # points_3d = points_4d[:3, :].T
    # print("3D Points:\n", points_3d)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    print("Fundamental matrix:\n", F)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # Cameras centre
    cpoints = np.vstack([M_l[:, -1].T, M_r[:, -1].T])
    Xs, Ys, Zs = cpoints.T
    ax.scatter3D(Xs, Ys, Zs, c=None, color='red')
    # Keipoints
    kpoints = points[mask]
    Xs, Ys, Zs = kpoints.T
    ax.scatter3D(Xs, Ys, Zs, c=None, color='blue')
    # Plot
    plt.show()
