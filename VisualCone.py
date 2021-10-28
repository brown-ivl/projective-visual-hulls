import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
import cv2 as cv

eps = 0.000001


class VisualCone:
    def __init__(self, intrinsic_matrix, extrinsic_matrix, silhouette):
        self.projection_matrix = intrinsic_matrix.dot(extrinsic_matrix)

        self.outline = self.get_outline(silhouette)
        self.silhouette = silhouette
        self.camera_location = self.get_camera_location(extrinsic_matrix)
        self.xyzs = self.get_xyz_coordinates(
            self.outline, self.projection_matrix)

    @staticmethod
    def get_camera_location(extrinsic_matrix):
        rotation = extrinsic_matrix[:, :3]
        translation = extrinsic_matrix[:, 3]

        return (-rotation.T).dot(translation)

    @staticmethod
    def get_outline(silhouette):
        '''Return contour points counter-clockwisely.'''
        silhouette = silhouette.astype(np.uint8)
        contours, _ = cv.findContours(
            silhouette, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        outline = np.squeeze(contours[0])

        return outline

    @staticmethod
    def get_xyz_coordinates(outline, projection_matrix, num=100):

        # choices = np.random.choice(len(outline), num)
        # outline = outline[choices]
        uvs = list(
            np.stack((outline[:, 0], outline[:, 1], np.ones(len(outline[:, 0]))), axis=-1))

        p_inv = lin.pinv(projection_matrix)

        points = map(p_inv.dot, uvs)
        points = np.array([point[:3] / point[3]
                          for point in points], dtype=np.float64)

        return points
