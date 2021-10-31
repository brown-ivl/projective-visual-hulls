import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import re
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
    # fig, ax = plt.subplots()

    # may have bugs
    projection = []
    for segment in branches:
        u0 = segment[0][0]
        v0 = segment[1][0]
        u1 = segment[0][1]
        v1 = segment[1][1]
        if u0 is not None and v0 is not None and u1 is not None and v1 is not None:
        
            xi = np.append(outline_i[u0], 1).reshape(-1, 1)
            xj = np.append(outline_j[v0], 1).reshape(-1, 1)

            xk = np.squeeze(np.sign(np.cross(Fik.dot(xi).T, ekj.T))*np.cross(Fjk.dot(xj).T, Fik.dot(xi).T))
            x0 = np.array([abs(xk[0]/xk[2]), abs(xk[1]/xk[2])])
        
        
        
            xi = np.append(outline_i[u1], 1).reshape(-1, 1)
            xj = np.append(outline_j[v1], 1).reshape(-1, 1)

            xk = np.squeeze(np.sign(np.cross(Fik.dot(xi).T, ekj.T))*np.cross(Fjk.dot(xj).T, Fik.dot(xi).T))
            x1 = np.array([abs(xk[0]/xk[2]), abs(xk[1]/xk[2])])
        
        
            # ax.plot([x0[0], x1[0]], [x0[1], x1[1]], 'b')
            projection.append([[x0[0], x1[0]], [x0[1], x1[1]]])
        else:
            branches.remove(segment)

    # plt.imshow(cone_k.silhouette,'gray')
    # ax.set_xlim([0, 640])
    # ax.set_ylim([0, 480])
    
    # plt.show()
    # fig, ax = plt.subplots()
    clipped = []

    # may have bugs
    counter = 0
    for p in projection:
        x0, x1 = p[0]
        y0, y1 = p[1]
        if x0 < 640 and x1 < 640 and y0 < 480 and y1 < 480:
            
            if x0 >= 0 and x1 >= 0 and y0 >= 0 and y1 >=0:
                start = np.array([x0, y0, 1])
                end = np.array([x1 ,y1 ,1])
                if cone_k.silhouette[int(y0),int(x0)]==0 and cone_k.silhouette[int(y1),int(x1)]==0:
                    pass
                elif cone_k.silhouette[int(y0),int(x0)]!=0 and cone_k.silhouette[int(y1),int(x1)]!=0:
                    clipped.append(p)
                    clipped_branches.append(branches[counter])
                else:
                    pass
                    # l = np.cross(start, end)
                    # intersections_indices = utils.get_intersections_indices(l, outline_k, start)
                    # for w in intersections_indices:
                    #     if outline_k[w][0] < max(x0,x1) and outline_k[w][0] > min(x0,x1) and outline_k[w][1] < max(y0,y1) and outline_k[w][1] > min(y0,y1):
                    #         ws.append(w)
                    #         if intersections_indices[w] == False:
                    #             clipped.append([[outline_k[w][0], x1],[outline_k[w][1], y1]])
                    #         else:
                    #             clipped.append([[outline_k[w][0], x0],[outline_k[w][1], y0]])
        counter += 1
            
    # for p in clipped:
    #     ax.plot(p[0],p[1], 'b')
    # plt.imshow(cone_k.silhouette,'gray')
    # ax.set_xlim([0, 640])
    # ax.set_ylim([0, 480])
    
    # plt.show()
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
    