import glob
import numpy as np
import matplotlib.pyplot as plt
import utils
import time


def trace(cone_i, cone_j):
    # Now we only have two frames.
    Fij, eij, eji = utils.get_fmatrices_epipoles(
        cone_i.projection_matrix, cone_j.projection_matrix)

    # 4.1. Tracing an Intersection Curve
    # 4.1.1. Critical Points of an Intersection Curve.
    u, outline_i = utils.process_regular_parameterization(cone_i.outline)
    v, outline_j = utils.process_regular_parameterization(cone_j.outline)
    

    epipolar_tangencies_i = utils.get_epipolar_tangencies(
        outline_i, eij)
    epipolar_tangencies_j = utils.get_epipolar_tangencies(
        outline_j, eji)

    critical_points = utils.get_critical_points(
        outline_i, outline_j, epipolar_tangencies_i, epipolar_tangencies_j, Fij, eij, eji)
    # utils.display_views('gourd/frame_00000000_Color_00.png', outline_i, outline_j,
    #                     epipolar_tangencies_i, epipolar_tangencies_j, Fij, eij, critical_points)
    
    # 4.1.2. The Tracing Algorithm
    increment = 1

    branches = utils.trace_branches(critical_points,
                                    outline_i, outline_j, increment, Fij, eji)

    # print('end')

    # utils.plot_points_branches(branches, len(
    #     outline_i), len(outline_j), critical_points)
    # utils.display_3D_representation(branches, outline_i, outline_j, cone_i.projection_matrix, cone_j.projection_matrix)

    return branches

# 4.2 Finding the 1-skeleton
def clip(branches, cone_i, cone_j, cone_k):
    clipped_branches = []
    u, outline_i = utils.process_regular_parameterization(cone_i.outline)
    v, outline_j = utils.process_regular_parameterization(cone_j.outline)
    w, outline_k = utils.process_regular_parameterization(cone_k.outline)
    Fik, eik, eki = utils.get_fmatrices_epipoles(
        cone_i.projection_matrix, cone_k.projection_matrix)
    Fjk, ejk, ekj = utils.get_fmatrices_epipoles(
        cone_j.projection_matrix, cone_k.projection_matrix)

    projection = utils.oriented_epipolar_transfer(branches, Fik, Fjk, ekj, outline_i, outline_j)
    clipped_branches = utils.clip(projection, cone_k, branches)
    
    points = utils.display_3D_representation(clipped_branches, outline_i, outline_j, cone_i.projection_matrix, cone_j.projection_matrix)
        
    return points, clipped_branches

if __name__ == "__main__":
    timer0 = time.perf_counter()
    print("Creating cones...")
    inputs = zip(glob.glob('gourd/*.png'), glob.glob('gourd/*.json'))
    cones = list(map(utils.create_cone, sorted(inputs)))
    point_clouds = utils.get_pc(sorted(glob.glob('nocs/*.png')))
    # utils.display_cones(cones[:2], point_clouds, show_rays=True)

    all_points = []
    skeletons = []
    for i in range(1,len(cones)):
        
        for skeleton in skeletons:
            skeletons.remove(skeleton)
            branches, ii, jj = skeleton
            print('start clipping:', ii,jj,i)
            points, branches = clip(branches, cones[ii], cones[jj], cones[i])
            print('end clipping', ii,jj,i)
            skeletons.append([branches, ii, jj])
            if i == len(cones)-1:
                all_points += points

        for j in range(0, i):
            print('start tracing:', i,j)
            branches = trace(cones[i], cones[j])
            print('end tracing:', i,j)
            points = []
            for k in range(0, i):
                if k != j:
                    print('start clipping:', i,j,k)
                    points, branches = clip(branches, cones[i], cones[j], cones[k])
                    print('end clipping', i,j,k)
            skeletons.append([branches, i, j])
            if i == len(cones)-1:
                all_points += points
    


    timer1 = time.perf_counter()
    time = timer1- timer0
    print('spend time: ', time,'s')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for p in all_points:
        ax.plot(p[0], p[1],
                p[2], color='black')
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    plt.show()
    