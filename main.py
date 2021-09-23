import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import re
import utils


if __name__ == "__main__":

    print("Creating cones...")
    inputs = zip(glob.glob('gourd/*.png'), glob.glob('gourd/*.json'))
    cones = list(map(utils.create_cone, sorted(inputs)))
    point_clouds = utils.get_pc(sorted(glob.glob('nocs/*.png')))
    # utils.display_cones(cones[:2], point_clouds, show_rays=True)
    # Now we only have two frames.
    Fij, eij, eji = utils.get_fmatrices_epipoles(
        cones[0].projection_matrix, cones[1].projection_matrix)

    # 4.1. Tracing an Intersection Curve
    # 4.1.1. Critical Points of an Intersection Curve.
    vertices_i, outline_i = utils.get_polygon(cones[0].outline)
    vertices_j, outline_j = utils.get_polygon(cones[1].outline)

    epipolar_tangencies_i = utils.get_epipolar_tangencies(
        vertices_i, eij)
    epipolar_tangencies_j = utils.get_epipolar_tangencies(
        vertices_j, eji)

    critical_points = utils.get_critical_points(
        outline_i, outline_j, epipolar_tangencies_i, epipolar_tangencies_j, Fij, eij, eji)
    utils.display_views('gourd/frame_00000000_Color_00.png', outline_i, outline_j,
                        epipolar_tangencies_i, epipolar_tangencies_j, Fij, eij, critical_points, vertices_i)
    utils.display_views('gourd/frame_00000001_Color_00.png', outline_j, outline_i,
                        epipolar_tangencies_j, epipolar_tangencies_i, Fij, eji, critical_points, vertices_j)
    # 4.1.2. The Tracing Algorithm
    increment = 10

    branches = utils.trace_branches(critical_points,
                                    outline_i, outline_j, increment, Fij, eji)

    print('end')

    utils.plot_points_branches(branches, len(
        outline_i), len(outline_j), critical_points)
