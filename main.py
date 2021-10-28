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
    
    def trace(cone_i, cone_j):
        # Now we only have two frames.
        Fij, eij, eji = utils.get_fmatrices_epipoles(
            cone_i.projection_matrix, cone_j.projection_matrix)

        # 4.1. Tracing an Intersection Curve
        # 4.1.1. Critical Points of an Intersection Curve.
        u, outline_i = utils.process_regular_parameterization(cone_i.outline)
        v, outline_j = utils.process_regular_parameterization(cone_j.outline)
        # w, outline_k = utils.process_regular_parameterization(cones[2].outline)

        epipolar_tangencies_i = utils.get_epipolar_tangencies(
            outline_i, eij)
        epipolar_tangencies_j = utils.get_epipolar_tangencies(
            outline_j, eji)

        critical_points = utils.get_critical_points(
            outline_i, outline_j, epipolar_tangencies_i, epipolar_tangencies_j, Fij, eij, eji)
        # utils.display_views('gourd/frame_00000000_Color_00.png', outline_i, outline_j,
        #                     epipolar_tangencies_i, epipolar_tangencies_j, Fij, eij, critical_points)
        # utils.display_views('gourd/frame_00000001_Color_00.png', outline_j, outline_i,
        #                     epipolar_tangencies_j, epipolar_tangencies_i, Fij, eji, critical_points)
        # 4.1.2. The Tracing Algorithm
        increment = 1

        branches = utils.trace_branches(critical_points,
                                        outline_i, outline_j, increment, Fij, eji)

        print('end')

        utils.plot_points_branches(branches, len(
            outline_i), len(outline_j), critical_points)
        utils.display_3D_representation(branches, outline_i, outline_j, cone_i.projection_matrix, cone_j.projection_matrix)

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
        fig, ax = plt.subplots()
        projection = []
        for segment in branches:
            u0 = segment[0][0]
            v0 = segment[1][0]
            u1 = segment[0][1]
            v1 = segment[1][1]
            if u0 and v0 and u1 and v1:
            
                xi = np.append(outline_i[u0], 1).reshape(-1, 1)
                xj = np.append(outline_j[v0], 1).reshape(-1, 1)

                xk = np.squeeze(np.sign(np.cross(Fik.dot(xi).T, ekj.T))*np.cross(Fjk.dot(xj).T, Fik.dot(xi).T))
                x0 = np.array([xk[0]/xk[2], xk[1]/xk[2]])
            
            
            
                xi = np.append(outline_i[u1], 1).reshape(-1, 1)
                xj = np.append(outline_j[v1], 1).reshape(-1, 1)

                xk = np.squeeze(np.sign(np.cross(Fik.dot(xi).T, ekj.T))*np.cross(Fjk.dot(xj).T, Fik.dot(xi).T))
                x1 = np.array([xk[0]/xk[2], xk[1]/xk[2]])
            
            
                ax.plot([-x0[0], -x1[0]], [x0[1], x1[1]], 'b')
                projection.append([[-x0[0], -x1[0]], [x0[1], x1[1]]])
            else:
                branches.remove(segment)

        plt.imshow(cone_k.silhouette,'gray')
        ax.set_xlim([0, 640])
        ax.set_ylim([0, 480])
        
        plt.show()
        fig, ax = plt.subplots()
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
                
        for p in clipped:
            ax.plot(p[0],p[1], 'b')
        plt.imshow(cone_k.silhouette,'gray')
        ax.set_xlim([0, 640])
        ax.set_ylim([0, 480])
        
        plt.show()
        points = utils.display_3D_representation(clipped_branches, outline_i, outline_j, cone_i.projection_matrix, cone_j.projection_matrix)
        
        return points

    
    all_points = []
    branches = []
    for i in range(1,len(cones)):
        if len(branches) != 0:
            points = clip(branches, cones[i-1], cones[j], cones[i])
            all_points += points
            print(i,j,k)
        for j in range(0, i):
            branches = trace(cones[i], cones[j])
            for k in range(0, i):
                if k != j:
                    points = clip(branches, cones[i], cones[j], cones[k])
                    all_points += points
                    print(i,j,k)
    fig = plt.figure()
    
    ax = fig.add_subplot(projection='3d')
    for p in all_points:
        ax.plot(p[0], p[1],
                p[2], color='black')
    
    plt.show()