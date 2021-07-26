import numpy as np
import scipy.linalg as lin
from skimage.segmentation import find_boundaries
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sys
from itertools import product

from tqdm import tqdm
import functools
import itertools

eps = 0.000001


class VisualCone:
    def __init__(self, intrinsic_matrix, extrinsic_matrix, silhouette):
        projection_matrix = intrinsic_matrix.dot(extrinsic_matrix)

        outline = self.generate_outline(silhouette)

        self.camera_location = self.get_camera_location(extrinsic_matrix)
        self.xyzs = self.get_xyz_coordinates(outline, projection_matrix)

    @staticmethod
    def get_ray_intersection(a0, b0, a1, b1):
        # compute unit vectors of directions of lines A and B
        vector_a = (a1 - a0) / np.linalg.norm(a1 - a0)
        vector_b = (b1 - b0) / np.linalg.norm(b1 - b0)

        # find unit direction vector for line C, which is perpendicular to lines A and B
        vector_c = np.cross(vector_b, vector_a);
        vector_c /= np.linalg.norm(vector_c)

        # solve the system derived in user2255770's answer from StackExchange: https://math.stackexchange.com/q/1993990
        RHS = b0 - a0
        LHS = np.array([vector_a, -vector_b, vector_c]).T
        t = np.linalg.solve(LHS, RHS)

        dist = t[2] / np.linalg.norm(vector_c)
        QA = a0 + (t[0] * (vector_a))
        QB = b0 + (t[1] * (vector_b))

        return dist, QA, QB

    def get_cone_intersection(self, other, min_dist_threshold=0.00001):
        XA0 = self.camera_location
        XB0 = other.camera_location
        intersection_points = []
        for XA1 in tqdm(self.xyzs):
            distances = {}
            for XB1 in other.xyzs:
                dist, QA, QB = self.get_ray_intersection(XA0, XB0, XA1, XB1)
                distances[dist] = QA

            sorted_distances = sorted(distances)
            if sorted_distances[0] < min_dist_threshold:
                intersection_points.append(distances[sorted_distances[0]])
            if sorted_distances[1] < min_dist_threshold:
                intersection_points.append(distances[sorted_distances[1]])

        return intersection_points

    @staticmethod
    def is_zero(val):
        return abs(val) <= sys.float_info.epsilon

    def display_cone(self):
        print('Displaying cone...')

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        print("Plotting rays...")
        for i, xyz in enumerate(self.xyzs):
            print(f"Plotting ray {i}/{len(self.xyzs)}")
            ax.plot([self.camera_location[0], xyz[0]], [self.camera_location[1], xyz[1]],
                    zs=[self.camera_location[2], xyz[2]])

        print("Plotting points...")
        ax.scatter(self.xyzs[:, 0], self.xyzs[:, 1], self.xyzs[:, 2], c='r', marker='o')

        print("Plotting camera...")
        ax.scatter(self.camera_location[0], self.camera_location[1], self.camera_location[2], c='b', marker='o')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        print("Displaying plot!")
        plt.show()

    @staticmethod
    def get_camera_location(extrinsic_matrix):
        rotation = extrinsic_matrix[:, :3]
        translation = extrinsic_matrix[:, 3]

        return (-rotation.T).dot(translation)

    @staticmethod
    def generate_outline(silhouette):
        outlines = find_boundaries(silhouette)
        outlines = outlines.astype(float)

        return outlines

    @staticmethod
    def get_xyz_coordinates(outline, projection_matrix):
        indices = np.argwhere(outline == 1)
        uvs = list(np.stack((indices[:, 0], indices[:, 1], np.ones(len(indices[:, 0]))), axis=-1))

        p_inv = lin.pinv(projection_matrix)

        points = map(p_inv.dot, uvs)
        points = np.array([point[:3] / point[3] for point in points], dtype=np.float64)

        return points
