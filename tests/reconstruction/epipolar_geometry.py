# -*- coding: utf-8 -*-
import cv2
import numpy as np
from math import cos, sin, pi
from matplotlib import pyplot as plt


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


class PointsReconstructor():
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.keypoints = None
        self.pts_cameras = None
        self.pts_objects = None
        self.good = None
        self.kp = None

        self.Kinv = np.eye(3)

    def _extract_features(self, frames):
        img1, img2 = frames[:2]

        if img1 is None or img2 is None:
            print("No image!")
            return

        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

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
        self.good = good
        self.kp = [kp1, kp2]

        return [pts1, pts2]

    def _camera_params(self, fx, fy, sx, sy, cx, cy, cam_t=np.array([0., 0., 0.]), cam_R=np.array([0., 0., 0.])):

        """
        fx, fy = (1.0, 1.0)  # Distância focal
        sx, sy = img1.shape[:2][::-1]  # Escala da projeção por eixo
        cx, cy = (sx//2, sy//2)  # Centro óptico no plano de projeção
        """

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
        R = np.dot(np.dot(M.z_rotation(cam_R[2]), M.y_rotation(cam_R[1])), M.x_rotation(cam_R[0]))
        t = M.translation(cam_t[0], cam_t[1], cam_t[2])
        Rt = np.dot(R, t)

        # Matriz calibrada de transformação dos pontos
        P = np.dot(K, np.dot(Pi, Rt))
        print("camera shape", P.shape)

        self.Kinv = Kinv

    def _normalize_points(self, points):
        return np.dot(self.Kinv, np.concatenate([points, np.ones((points.shape[0], 1))], axis=1).T).T[:, 0:2]

    def calibrate(self, frames=[None]):
        if None in [x for x in frames if x is None]:
            print("No image!")
            return

        # Extrai keypoints
        img1, img2 = frames[:2]
        height, width = img1.shape[:2]
        self.keypoints = self._extract_features(frames=frames)

        # Parâmetros intrinsecos e extrinsecos da câmera
        self._camera_params(fx=1, fy=1, sx=width, sy=height, cx=width/2, cy=height/2)
        camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        if self.keypoints is None:
            print("No keypoints!")
            return
        pts1, pts2 = self.keypoints

        # using the essential matrix can get you the rotation/translation bet. cameras,
        # although there are two possible rotations:
        E, m2 = cv2.findEssentialMat(pts1, pts2, camera_matrix, cv2.RANSAC, 0.999, 1.0)
        print("Essential matrix:\n", E)

        # recoverPose gets you an unambiguous R and t. One of the R's above does agree with the R determined here.
        # RecoverPose can already triangulate, I check by hand below to compare results.
        pts1_norm = self._normalize_points(points=pts1)
        pts2_norm = self._normalize_points(points=pts2)
        retval, R, t, mask2, triangulatedPoints = cv2.recoverPose(
            E, pts1_norm, pts2_norm, camera_matrix, distanceThresh=0.5)
        print("Recovered pose:\n", R, t)

        # given R,t you can  explicitly find 3d locations using projection
        self.M_r = np.concatenate((R, t), axis=1)
        self.M_l = np.concatenate((np.eye(3, 3), np.zeros((3, 1))), axis=1)
        self.proj_r = np.dot(camera_matrix, self.M_r)
        self.proj_l = np.dot(camera_matrix, self.M_l)

        self.pts_cameras = np.vstack([self.M_l[:, -1].T, self.M_r[:, -1].T])

    def _triangulate_normalized_points(self, pose_1w, pose_2w, kpn_1, kpn_2):
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

    def reconstruct(self, points):
        pts1, pts2 = points[:2]
        pts1_norm = self._normalize_points(points=pts1)
        pts2_norm = self._normalize_points(points=pts2)
        points, mask = self._triangulate_normalized_points(
            pose_1w=self.proj_l, pose_2w=self.proj_r,
            kpn_1=pts1_norm, kpn_2=pts2_norm)

        self.pts_objects = points[mask]

    # Cria setas para referência dos eixos, rgb(xyz)
    # y e z invertidos para representação comum
    def _set_arrows_orientation(self, ax, origin=np.array([0., 0., 0.]),
                                vector=np.append(np.eye(3), np.ones((1, 3)), axis=0)):
        x, y, z = origin
        xv, yv, zv = vector[:-1, 0]
        xs, ys, zs = xv-x, yv-y, zv-z
        ax.quiver(
            x, z, y,
            xs, zs, ys,
            color="red", alpha=1., lw=1
        )
        xv, yv, zv = vector[:-1, 1]
        xs, ys, zs = xv-x, yv-y, zv-z
        ax.quiver(
            x, z, y,
            xs, zs, ys,
            color="green", alpha=1., lw=1
        )
        xv, yv, zv = vector[:-1, 2]
        xs, ys, zs = xv-x, yv-y, zv-z
        ax.quiver(
            x, z, y,
            xs, zs, ys,
            color="blue", alpha=1., lw=1
        )

    def render_scene(self):
        _ = plt.figure()
        ax = plt.axes(projection='3d')

        # Cameras
        if self.pts_cameras is not None:
            x0, z0, y0 = self.pts_cameras[0]
            ax.scatter3D(x0, y0, z0, c=None, color='red')
            x1, z1, y1 = self.pts_cameras[1]
            ax.scatter3D(x1, y1, z1, c=None, color='blue')

            # Arrows
            # self._set_arrows_orientation(ax=ax)
            # self._set_arrows_orientation(
            #     ax=ax, origin=self.pts_cameras[1],
            #     vector=np.append(self.M_r[:3, :3], [self.M_r[:, 3]], axis=0))

        # Keipoints
        if self.pts_objects is not None:
            Xs, Zs, Ys = self.pts_objects.T
            ax.scatter3D(Xs, Ys, Zs, c=None, color='green')

        # Plot
        plt.show()

    def match_keypoints_homography(self, frames):
        img1, img2 = frames[:2]
        src_pts, dst_pts = self.keypoints[:2]

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        draw_params = dict(
            matchColor=(0, 255, 0),  # draw matches in green color
            singlePointColor=None,
            matchesMask=matchesMask,  # draw only inliers
            flags=2)

        img3 = cv2.drawMatches(img1, self.kp[0], img2, self.kp[1], self.good, None, **draw_params)
        cv2.imshow('Match keypoints', img3)
        cv2.waitKey()

    def match_keypoints_fundamental(self, frames):
        img1, img2 = frames[:2]
        pts1, pts2 = self.keypoints[:2]
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        print("Fundamental matrix:\n", F)

        # We select only inlier points
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = self._drawlines(img1, img2, lines1, pts1, pts2)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = self._drawlines(img2, img1, lines2, pts2, pts1)

        plt.subplot(121), plt.imshow(img5)
        plt.subplot(122), plt.imshow(img3)
        plt.show()

    def _drawlines(self, img1, img2, lines, pts1, pts2):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r, c = img1.shape
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)

        return img1, img2
