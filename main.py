from VisualCone import VisualCone
import glob
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import re
import json
import config as c
import utils



def get_silhouette(img_file):
    img = cv.imread(img_file,0)
    _, silhouette = cv.threshold(img,254.9,255,cv.THRESH_BINARY)
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
    x,y,z,w = camera_pose['rotation'].values()

    Q = np.array([[1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
                  [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
                  [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])

    C = np.array([[p_x], [p_y], [p_z]])

    R = Q.T
    t = -np.dot(R,C)
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


if __name__ == "__main__":

    #    print(intersection(*[Line3D((-2,-1.0,0.0), ()), Line3D((),()]))
    print("Creating cones...")
    inputs = zip(glob.glob('shapenet/*.png'), glob.glob('shapenet/*.json'))
    cones = list(map(create_cone, sorted(inputs)))
    point_clouds = utils.get_pc(sorted(glob.glob('nocs/*.png')))
    utils.display_cones(cones, point_clouds, show_rays=False)
    # TODO Below part is not completed.
    intersections = []
   
    for cone1_num in range(len(cones)):
        for cone2_num in range(cone1_num+1,len(cones)):
            print(f"Computing intersections between cone {cone1_num} and cone {cone2_num}")
            # cones[cone2_num].display_cone()
            
            intersections += cones[cone1_num].get_cone_intersection(cones[cone2_num])
    plot_points(intersections)