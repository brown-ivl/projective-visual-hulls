import matplotlib.pyplot as plt
from numpy.lib.function_base import append
from tk3dv.nocstools import datastructures as ds
from skimage.draw import line
import numpy as np
import cv2 as cv
import json
import config as c
from VisualCone import VisualCone
from scipy.interpolate import splprep, splev


def get_silhouette(img_file):
    img = cv.imread(img_file, 0)
    _, silhouette = cv.threshold(img, 254.9, 255, cv.THRESH_BINARY_INV)
    silhouette = cv.flip(silhouette, 0)
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


def display_views(image_path, outline_i, outline_j, epipolar_tangencies_i, epipolar_tangencies_j, Fij, e, critical_points):
    image = plt.imread(image_path)[::-1]
    #TODO e position reversed
    e0 = e[0] / e[2]
    e1 = e[1] / e[2]

    ei = (e0[0], e1[0])
    
    plt.plot(outline_i[:,0], outline_i[:,1], 'bo', ms=1)
    
    for x in epipolar_tangencies_i:
        
        plt.plot(x[0],x[1], 'ro', ms=4)

        plt.axline(x, ei, linewidth=2)

    for _, x in enumerate(outline_j):
        if tuple(x) in epipolar_tangencies_j:
            lij = np.dot(Fij, np.append(x, 1).reshape(-1, 1)).T[0]
            slope = -lij[0]/lij[1]
            plt.axline(ei, slope=slope, c='r', ls='--')
    # for u, v in critical_points:
    #     cv.circle(image, list(outline_i.keys())[u], radius=0,
    #             color=(0, 0, 255), thickness=-1)
    #     plt.axline(ei, list(outline_i.keys())[u], c='r', ls='--',linewidth=1)
    plt.imshow(image, origin='lower')
    
    plt.show()


def display_3D_representation(branches, outline_i, outline_j, pi, pj):
    # Use numpy.linalg.lstsq(A, B).
    points = []
    A = np.vstack((pi, pj))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for segment in branches:
        u0 = segment[0][0]
        v0 = segment[1][0]
        u1 = segment[0][1]
        v1 = segment[1][1]
        if u0 and v0 and u1 and v1:
            xi = np.append(outline_i[u0], 1).reshape(-1, 1)
            xj = np.append(outline_j[v0], 1).reshape(-1, 1)

            B = np.vstack((xi, xj))
            r = np.linalg.lstsq(A, B)[0]
            X0 = np.array([r[0]/r[3], r[1]/r[3], r[2]/r[3]])

        
        
            xi = np.append(outline_i[u1], 1).reshape(-1, 1)
            xj = np.append(outline_j[v1], 1).reshape(-1, 1)

            B = np.vstack((xi, xj))
            r = np.linalg.lstsq(A, B)[0]
            X1 = np.array([r[0]/r[3], r[1]/r[3], r[2]/r[3]])

            ax.plot([X0[0][0], X1[0][0]], [X0[1][0], X1[1][0]],
            [X0[2][0], X1[2][0]], color='black')
            points.append([[X0[0][0], X1[0][0]], [X0[1][0], X1[1][0]],[X0[2][0], X1[2][0]]])

    # ax.set_xlim([-2, 2])
    # ax.set_ylim([-2, 2])
    # ax.set_zlim([-2, 2])

    plt.show()
    return points


def create_cone(inputs):
    img_file, parameter_file = inputs
    img = get_silhouette(img_file)
    # plt.imshow(img, cmap="gray", origin='lower')
    
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
                    [np.linalg.det(np.hstack((Qi, Ri, Rj, Pj))), np.linalg.det(
                        np.hstack((Ri, Pi, Rj, Pj))), np.linalg.det(np.hstack((Pi, Qi, Rj, Pj)))],
                    [np.linalg.det(np.hstack((Qi, Ri, Pj, Qj))), np.linalg.det(np.hstack((Ri, Pi, Pj, Qj))), np.linalg.det(np.hstack((Pi, Qi, Pj, Qj)))]])

    eij = np.array([np.linalg.det(np.hstack((Pi, Pj, Qj, Rj))), np.linalg.det(np.hstack(
        (Qi, Pj, Qj, Rj))), np.linalg.det(np.hstack((Ri, Pj, Qj, Rj)))]).reshape(-1, 1)
    eji = np.array([np.linalg.det(np.hstack((Pj, Pi, Qi, Ri))), np.linalg.det(np.hstack(
        (Qj, Pi, Qi, Ri))), np.linalg.det(np.hstack((Rj, Pi, Qi, Ri)))]).reshape(-1, 1)

    return Fij, eij, eji


def process_regular_parameterization(outline):
    
    tck, u = splprep([outline[:,0], outline[:,1]], s=30, per=3)
    u = np.arange(0,1,1/500)
    parameterized_outline = splev(u, tck)
    # fig, ax = plt.subplots()
    # ax.plot(parameterized_outline[0], parameterized_outline[1], 'ro')
    # ax.set_xlim([0, 640])
    # ax.set_ylim([0, 480])
    # plt.show()
    parameterized_outline = np.array(parameterized_outline).T
    return u, parameterized_outline

def get_tangent(x_previous, x):

    tangent = np.cross(np.append(x_previous, 1), np.append(x, 1))

    return tangent


def get_epipolar_tangencies(outline, e):

    epipolar_tangencies = {}
    for i, x in enumerate(outline):
        if i + 1 >= len(outline):
            i = -1
        tangent_previous = get_tangent(outline[i-1], x)
        tangent_next = get_tangent(x, outline[i+1])

        a = tangent_previous.dot(e)
        b = tangent_next.dot(e)

        # Is this an epipolar tangency?
        if np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) == -1.0:

            # fuu > 0?
            if a < 0 and b > 0:
                epipolar_tangencies[tuple(x)] = True
            if a > 0 and b < 0:
                epipolar_tangencies[tuple(x)] = False

    return epipolar_tangencies


def get_intersections_indices(l, outline, e):
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    intersections_indices = {}
    
    for i, x in enumerate(outline):
        a = np.append(outline[i-1],1).dot(l)
        b = np.append(x,1).dot(l)
        if np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)) == -1.0:
        
            # fv > 0
            if np.dot(get_tangent(outline[i-1], x), e) > 0:
                intersections_indices[i] = True
            if np.dot(get_tangent(outline[i-1], x), e) < 0:
                intersections_indices[i] = False
            

    return intersections_indices


def get_critical_points(outline_i, outline_j, epipolar_tangencies_i, epipolar_tangencies_j, Fij, eij, eji):
    critical_points = {}
    for u, x in enumerate(outline_i):
        if tuple(x) in epipolar_tangencies_i:
            lji = np.dot(Fij, np.append(x, 1).reshape(-1, 1))
            intersections_indices_j = get_intersections_indices(
                lji, outline_j, eji)
            for v in intersections_indices_j:
                # (u0, v0) is a local maximum (resp. minimum) in the v-direction if the signs of fv and fuu are the same (resp.opposite).
                if intersections_indices_j[v] == epipolar_tangencies_i[tuple(x)]:
                    critical_points[(u, v)] = '2B'  # local max
                else:
                    critical_points[(u, v)] = '2A'  # local min
    for v, x in enumerate(outline_j):
        if tuple(x) in epipolar_tangencies_j:
            lij = np.dot(Fij, np.append(x, 1).reshape(-1, 1))
            intersections_indices_i = get_intersections_indices(
                lij, outline_i, eij)
            for u in intersections_indices_i:
                # (u0, v0) is a local maximum (resp. minimum) in the u-direction if the signs of fu and fvv are the same (resp.opposite).
                if intersections_indices_i[u] == epipolar_tangencies_j[tuple(x)]:
                    critical_points[(u, v)] = '3B'  # local max
                else:
                    critical_points[(u, v)] = '3A'  # local min
    return critical_points


def encounter_critical_points(label, critical_points, u, v, previous_u, previous_v):
    points = []

    for uu, vv in critical_points:
        if uu >= previous_u:
            if label == '++':
                if critical_points[(uu, vv)] == '2B':
                    if uu > previous_u and uu <= u:
                        points.append((uu, vv))
                elif critical_points[(uu, vv)] == '3B':
                    if vv > previous_v and vv <= v:
                        points.append((uu, vv))
            elif label == '+-':
                if critical_points[(uu, vv)] == '2A':
                    if uu > previous_u and uu <= u:
                        points.append((uu, vv))
                elif critical_points[(uu, vv)] == '3B':
                    if vv < previous_v and vv >= v:
                        points.append((uu, vv))

    if len(points) == 0:
        first_critical_point = None
    else:
        first_u = min([point[0] for point in points])
        points = sorted([point for point in points if point[0]
                        == first_u], key=lambda point: point[1])

        if label == '++':
            list = [point for point in points if point[1] >= previous_v]
            if len(list) == 0:
                first_critical_point = None
            else:
                first_critical_point = list[0]
        elif label == '+-':
            list = [point for point in points if point[1] <= previous_v]
            if len(list) == 0:
                first_critical_point = None
            else:
                first_critical_point = list[-1]

    # print(first_critical_point)
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
    v = None
    if len(intersections_indices_j) != 0:

        vs_larger = [v for v in intersections_indices_j if v >= previous_v]
        vs_smaller = [v for v in intersections_indices_j if v <= previous_v]
        if branch_labels == '++':

            if len(vs_larger) == 0:

                v = v_range-1
            else:
                v = min(vs_larger)

        else:

            if len(vs_smaller) == 0:

                v = 0
            else:
                v = max(vs_smaller)

    return v


def trace_branch(label, start_critical_point, critical_points, outline_i, outline_j, increment, Fij, eji):
    branches = []
    u, v = start_critical_point
    while u < len(outline_i)-1:

        previous_u = u

        previous_v = v
        if label == '++':
            if previous_v == len(outline_j)-1:
                previous_v = 0
        else:
            if previous_v == 0:
                previous_v = len(outline_j)-1

        u = u + increment

        if u > len(outline_i)-1:
            u = len(outline_i)-1

        lji = np.dot(Fij, np.append(
            outline_i[u], 1).reshape(-1, 1))
        intersections_indices_j = get_intersections_indices(
            lji, outline_j, eji)

        v = get_nearest_v(
            label, intersections_indices_j, previous_v, len(outline_j))
        if v is not None:

            first_critical_point = encounter_critical_points(
                label, critical_points, u, v, previous_u, previous_v)

            if first_critical_point is None:
                branches.append([[previous_u, u], [previous_v, v]])

                if u == len(outline_i)-1:

                    u = 0
            else:

                u, v = first_critical_point
                branches.append([[previous_u, u], [previous_v, v]])

                break
        else:
            if label == '++':
                first_critical_point = encounter_critical_points(
                    label, critical_points, u, len(outline_j), previous_u, previous_v)
            else:
                first_critical_point = encounter_critical_points(
                    label, critical_points, u, 0, previous_u, previous_v)

            if first_critical_point is None:
                branches.append([[previous_u, u], [previous_v, v]])
                if u == len(outline_i)-1:

                    u = 0
            else:

                u, v = first_critical_point
                branches.append([[previous_u, u], [previous_v, v]])
                break
            break
    return branches


def trace_branches(critical_points, outline_i, outline_j, increment, Fij, eji):

    branches = []

    for critical_point in critical_points:
        if critical_points[critical_point] == '2A' or critical_points[critical_point] == '3A':
            branches += trace_branch('++', critical_point, critical_points,
                                     outline_i, outline_j, increment, Fij, eji)
        if critical_points[critical_point] == '2B' or critical_points[critical_point] == '3A':
            branches += trace_branch('+-', critical_point, critical_points,
                                     outline_i, outline_j, increment, Fij, eji)

    return branches


def projecting(X, projection_matrix):
    X = np.append(X, 1)
    x = np.dot(projection_matrix, X.reshape(-1, 1))
    return x


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
