import matplotlib.pyplot as plt
from tk3dv.nocstools import datastructures as ds
from skimage.draw import line
import numpy as np
import cv2 as cv
import json
import config as c
from VisualCone import VisualCone


def get_silhouette(img_file):
    img = cv.imread(img_file, 0)
    _, silhouette = cv.threshold(img, 254.9, 255, cv.THRESH_BINARY_INV)
    return silhouette


def get_camera_pose(parameter_file):
    f = open(parameter_file)
    camera_pose = json.load(f)
    return camera_pose


def get_intrinsic_matrix():
    return np.array([[c.focal_length_x, c.axis_skew, c.camera_center_x],
                     [0, c.focal_length_y, c.camera_center_y],
                     [0, 0, 1]])


def get_extrinsic_matrix(camera_pose):
    p_x, p_y, p_z = camera_pose['position'].values()
    x, y, z, w = camera_pose['rotation'].values()

    Q = np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                  [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                  [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])

    C = np.array([[p_x], [p_y], [p_z]])

    R = Q.T
    t = -np.dot(R, C)
    return np.hstack((R, t))


def plot_points(points):
    points = np.array(points)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    print("Plotting points...")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    print("Displaying plot!")
    plt.show()


def plot_points_branches(branches, u_range, v_range, critical_points):
    fig = plt.figure()
    ax = fig.add_subplot()

    print("Plotting points...")

    for segment in branches:

        ax.plot(segment[0], segment[1], c='b')
    for point in critical_points:
        if critical_points[point] == '2A':
            ax.scatter(point[0], point[1], c='black', marker='v')
        if critical_points[point] == '2B':
            ax.scatter(point[0], point[1], c='black', marker='^')
        if critical_points[point] == '3A':
            ax.scatter(point[0], point[1], c='black', marker='<')
        if critical_points[point] == '3B':
            ax.scatter(point[0], point[1], c='black', marker='>')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_xlim([0, u_range])
    ax.set_ylim([0, v_range])
    print("Displaying plot!")
    plt.show()


def create_cone(inputs):
    img_file, parameter_file = inputs
    img = get_silhouette(img_file)
    # plt.imshow(img, cmap="gray")
    # plt.show()
    img = cv.resize(img, (img.shape[1], img.shape[0]))
    img = np.asarray(img, dtype='float32')

    camera_pose = get_camera_pose(parameter_file)
    intrinsic_matrix = get_intrinsic_matrix()
    extrinsic_matrix = get_extrinsic_matrix(camera_pose)

    return VisualCone(intrinsic_matrix, extrinsic_matrix, img)


def get_fmatrices_epipoles(pi, pj):
    '''Appendix A'''
    Pi, Qi, Ri = pi[0].reshape(-1, 1), pi[1].reshape(-1,
                                                     1), pi[2].reshape(-1, 1)
    Pj, Qj, Rj = pj[0].reshape(-1, 1), pj[1].reshape(-1,
                                                     1), pj[2].reshape(-1, 1)

    Fij = np.array([[np.linalg.det(np.hstack((Qi, Ri, Qj, Rj))), np.linalg.det(np.hstack((Ri, Pi, Qj, Rj))), np.linalg.det(np.hstack((Pi, Qi, Qj, Rj)))],
                    [np.linalg.det(np.hstack((Qi, Ri, Rj, Pj))), np.linalg.det(np.hstack((Ri, Pi, Rj, Pj))), np.linalg.det(np.hstack((Pi, Qi, Rj, Pj)))],
                    [np.linalg.det(np.hstack((Qi, Ri, Pj, Qj))), np.linalg.det(np.hstack((Ri, Pi, Pj, Qj))), np.linalg.det(np.hstack((Pi, Qi, Pj, Qj)))]])

    eij = np.array([np.linalg.det(np.hstack((Pi, Pj, Qj, Rj))), np.linalg.det(np.hstack(
        (Qi, Pj, Qj, Rj))), np.linalg.det(np.hstack((Ri, Pj, Qj, Rj)))]).reshape(-1, 1)
    eji = np.array([np.linalg.det(np.hstack((Pj, Pi, Qi, Ri))), np.linalg.det(np.hstack(
        (Qj, Pi, Qi, Ri))), np.linalg.det(np.hstack((Rj, Pi, Qi, Ri)))]).reshape(-1, 1)

    return Fij, eij, eji


def get_polygon(outline):
    epsilon = 0.002*cv.arcLength(outline, True)
    vertices = cv.approxPolyDP(outline, epsilon, True)
    vertices = np.squeeze(vertices)
    polygon_outline = {}
    for i, x in enumerate(vertices):
        discrete_line = list(zip(*line(*vertices[i-1], *x)))
        tengent = get_tengent(vertices[i-1], x)

        for p in discrete_line[:-1]:
            polygon_outline[p] = tengent
    return vertices, polygon_outline


def get_tengent(x_previous, x):

    tengent = np.cross(np.append(x_previous, 1), np.append(x, 1))

    return tengent


def get_frontier_points(vertices, e):

    frontier_points = {}
    for i, x in enumerate(vertices):
        if i + 1 >= len(vertices):
            i = -1
        tengent_previous = get_tengent(vertices[i-1], x)
        tengent_next = get_tengent(x, vertices[i+1])

        a = tengent_previous.dot(e)
        b = tengent_next.dot(e)

        # is a frontier point?
        if np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) == -1.0:

            # fuu > 0?
            if a < 0 and b > 0:
                frontier_points[tuple(x)] = True
            if a > 0 and b < 0:
                frontier_points[tuple(x)] = False

    

    return frontier_points


def get_intersections_indices(l, outline, e):

    intersections_indices = {}
    # TODO
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    # dist = 0.7071
    smallest_dist = 0.0
    tengent = None
    ii = None
    for i, x in enumerate(outline):
        dist = abs(np.dot(np.append(x,1), l))/np.sqrt(l[0]**2+l[1]**2)
        
        if dist < 0.71:
            if np.array_equal(tengent, outline[x]):
                if dist < smallest_dist:
                    intersections_indices.pop(ii)
                else:
                    continue
            
            if np.dot(outline[x], e) > 0:
                intersections_indices[i] = True
            if np.dot(outline[x], e) < 0:
                intersections_indices[i] = False
            smallest_dist = dist
            tengent = outline[x]
            ii = i

    return intersections_indices


def get_critical_points(outline_i, outline_j, frontier_points_i, frontier_points_j, Fij, eij, eji):
    critical_points = {}
    for u, x in enumerate(outline_i):
        if x in frontier_points_i:
            lji = np.dot(Fij, np.append(x, 1).reshape(-1, 1))
            intersections_indices_j = get_intersections_indices(
                lji, outline_j, eji)
            for v in intersections_indices_j:
                # (u0, v0) is a local maximum (resp. minimum) in the v-direction if the signs of fv and fuu are the same (resp.opposite).
                if intersections_indices_j[v] == frontier_points_i[x]:
                    critical_points[(u, v)] = '2B'  # local max
                else:
                    critical_points[(u, v)] = '2A'  # local min
    for v, x in enumerate(outline_j):
        if x in frontier_points_j:
            lij = np.dot(Fij, np.append(x, 1).reshape(-1, 1))
            intersections_indices_i = get_intersections_indices(
                lij, outline_i, eij)
            for u in intersections_indices_i:
                # (u0, v0) is a local maximum (resp. minimum) in the u-direction if the signs of fu and fvv are the same (resp.opposite).
                if intersections_indices_i[u] == frontier_points_j[x]:
                    critical_points[(u, v)] = '3B'  # local max
                else:
                    critical_points[(u, v)] = '3A'  # local min
    return critical_points


def contains_critical_points(label, critical_points, u, v, previous_u, previous_v):
    # TODO 
    first_critical_point = None
    
    for uu, vv in critical_points:
        if (label == '++' and critical_points[(uu, vv)] in ['2B', '3B'] and (vv > previous_v and uu > previous_u and (vv <= v or uu <= u))) \
                or (label == '+-' and critical_points[(uu, vv)] in ['2A', '3B'] and (vv < previous_v and uu > previous_u and (vv >= v or uu <= u))):

            # if first_critical_point is not None:
            #     dist = (uu-previous_u)**2 + \
            #         (vv-previous_v)**2  # uu - previous_u #
            #     # first_critical_point[0] - previous_u #
            #     dist_first = (
            #         first_critical_point[0]-previous_u)**2+(first_critical_point[1]-previous_v)**2
            #     if dist < dist_first:
            #         first_critical_point = (uu, vv)
            # else:
            #     first_critical_point = (uu, vv)
            dist = (uu-previous_u)**2 + (vv-previous_v)**2
            if dist < 1000:
                first_critical_point = (uu, vv)

    print(first_critical_point)
    return first_critical_point


def get_branch_labels(critical_points, u, v):
    labels = []
    if critical_points[(u, v)] == '2A' or critical_points[(u, v)] == '3A':
        labels.append('++')
    elif critical_points[(u, v)] == '2B' or critical_points[(u, v)] == '3A':
        labels.append('+-')
    else:
        pass
    return labels


def get_nearest_v(branch_labels, intersections_indices_j, previous_v, v_range):
    # TODO
    v = None
    if len(intersections_indices_j) != 0:
       
        vs_larger = [v for v in intersections_indices_j if v >= previous_v]
        vs_smaller = [v for v in intersections_indices_j if v <= previous_v]
        if branch_labels == '++':

            if len(vs_larger) == 0:
                # pass
                v = v_range-1
            else:
                v = min(vs_larger)

        else:

            if len(vs_smaller) == 0:
                # pass
                v = 0
            else:
                v = max(vs_smaller)

    return v


def trace_branch(label, start_critical_point, critical_points, i_outline, j_outline, increment, Fij, eji):
    branches = []
    u, v = start_critical_point
    while u < len(i_outline)-1:

        previous_u = u

        previous_v = v
        if label == '++':
            if previous_v == len(j_outline)-1:
                previous_v = 0
        else:
            if previous_v == 0:
                previous_v = len(j_outline)-1

        u = u + increment

        if u > len(i_outline)-1:
            u = len(i_outline)-1
        lji = np.dot(Fij, np.append(
            list(i_outline.keys())[u], 1).reshape(-1, 1))
        intersections_indices_j = get_intersections_indices(
            lji, j_outline, eji)

        v = get_nearest_v(
            label, intersections_indices_j, previous_v, len(j_outline))
        if v is not None:

            first_critical_point = contains_critical_points(
                label, critical_points, u, v, previous_u, previous_v)

            if first_critical_point is None:
                branches.append([[previous_u, u], [previous_v, v]])
                print(u, v)

                if u == len(i_outline)-1:

                    u = 0
            else:

                u, v = first_critical_point
                branches.append([[previous_u, u], [previous_v, v]])

                break
        else:
            if label == '++':
                first_critical_point = contains_critical_points(
                    label, critical_points, u, len(j_outline), previous_u, previous_v)
            else:
                first_critical_point = contains_critical_points(
                    label, critical_points, u, 0, previous_u, previous_v)

            if first_critical_point is None:
                branches.append([[previous_u, u], [previous_v, v]])
                print(u, v)
                if u == len(i_outline)-1:

                    u = 0
            else:

                u, v = first_critical_point
                branches.append([[previous_u, u], [previous_v, v]])
                break
            break
    return branches


def trace_branches(critical_points, i_outline, j_outline, increment, Fij, eji):

    branches = []

    for critical_point in critical_points:
        if critical_points[critical_point] == '2A' or critical_points[critical_point] == '3A':
            branches += trace_branch('++', critical_point, critical_points,
                                     i_outline, j_outline, increment, Fij, eji)
        if critical_points[critical_point] == '2B' or critical_points[critical_point] == '3A':
            branches += trace_branch('+-', critical_point, critical_points,
                                     i_outline, j_outline, increment, Fij, eji)

    # plot_points_branches(branch, len(cones[0].outline), len(cones[1].outline), critical_points)
    return branches


def projecting(X, projection_matrix):
    X = np.append(X, 1)
    x = np.dot(projection_matrix, X.reshape(-1, 1))
    return x

# def cal_partial_derivative(x, e):
#     f = np.linalg.det(np.hstack((x, np.gradient(x.T[0]).reshape(-1, 1), e)))

#     return f


def display_cones(cones, point_clouds, show_pc=False, show_rays=True):
    print('Displaying cones...')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for cone in cones:

        if show_rays:
            print("Plotting rays...")
            for i, xyz in enumerate(cone.xyzs):
                print(f"Plotting ray {i}/{len(cone.xyzs)}")
                ax.plot(xs=[cone.camera_location[0], xyz[0]], ys=[cone.camera_location[1], xyz[1]],
                        zs=[cone.camera_location[2], xyz[2]])

        print("Plotting points...")
        ax.scatter(cone.xyzs[:, 0], cone.xyzs[:, 1],
                   cone.xyzs[:, 2], c='r', marker='o', s=4)

        print("Plotting camera...")
        ax.scatter(cone.camera_location[0], cone.camera_location[1],
                   cone.camera_location[2], c='b', marker='o')

    if show_pc:
        scale = 1.0
        ax.scatter(scale*(point_clouds[:, 0]-0.5), scale *
                   (point_clouds[:, 1]-0.5), scale*(point_clouds[:, 2]-0.5), s=4)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    print("Displaying plot!")
    plt.show()


def nocs2pc(nocs_list, num):
    ''' Turns a tuple of NOCS maps into a combined point cloud '''
    nocs_pc = []
    for nocs_map in nocs_list:
        nocs = ds.NOCSMap(nocs_map)
        choices = np.random.choice(len(nocs.Points), num)
        point_clouds = nocs.Points[choices]
        nocs_pc.append(point_clouds)
    nocs_pc = np.concatenate(nocs_pc, axis=0)
    return nocs_pc


def get_pc(paths, num=100):
    nocs_map_list = []
    for path in paths:
        nocs_map = read_nocs_map(path)
        nocs_map_list.append(nocs_map)

    return nocs2pc(nocs_map_list, num)


def read_nocs_map(path):
    nocs_map = cv.imread(path, -1)
    nocs_map = nocs_map[:, :, :3]
    nocs_map = cv.cvtColor(nocs_map, cv.COLOR_BGR2RGB)
    return nocs_map
