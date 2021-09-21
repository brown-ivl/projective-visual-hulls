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

    frontier_points_i = utils.get_frontier_points(
        vertices_i, eij)
    frontier_points_j = utils.get_frontier_points(
        vertices_j, eji)
    
    def display_views(image_path, outline_i, outline_j, frontier_points_i, frontier_points_j, Fij, e, critical_points,vertices_i):   
        image = cv.imread(image_path)

        e[0] /= e[2]
        e[1] /= e[2]
        ei = (e[0][0], e[1][0])
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        for x in outline_i:
            cv.circle(image, x, radius=1,
                    color=(0, 0, 0), thickness=-1)
        # for x in vertices_i:
        #     cv.circle(image, x, radius=0,
        #             color=(225, 225, 0), thickness=-1)
        for x in frontier_points_i:
            cv.circle(image, x, radius=0,
                    color=(0, 0, 255), thickness=-1)

            plt.axline(x, ei,linewidth=2)
        
        
        for _, x in enumerate(outline_j):
            if x in frontier_points_j:
                lij = np.dot(Fij, np.append(x, 1).reshape(-1, 1)).T[0]
                slope = -lij[0]/lij[1]
                plt.axline(ei, slope=slope, c='r', ls='--')
        # for u, v in critical_points:
        #     cv.circle(image, list(outline_i.keys())[u], radius=0,
        #             color=(0, 0, 255), thickness=-1)
        #     plt.axline(ei, list(outline_i.keys())[u], c='r', ls='--',linewidth=1)
        plt.imshow(image)

        plt.show()

    
    critical_points = utils.get_critical_points(
        outline_i, outline_j, frontier_points_i, frontier_points_j, Fij, eij, eji)
    display_views('gourd/frame_00000000_Color_00.png', outline_i, outline_j, frontier_points_i, frontier_points_j, Fij, eij, critical_points,vertices_i)
    display_views('gourd/frame_00000001_Color_00.png', outline_j, outline_i, frontier_points_j, frontier_points_i, Fij, eji, critical_points,vertices_j)
    # 4.1.2. The Tracing Algorithm
    increment = 10

    branches = utils.trace_branches(critical_points,
                                    outline_i, outline_j, increment, Fij, eji)

    print('end')

    utils.plot_points_branches(branches, len(
        outline_i), len(outline_j), critical_points)
