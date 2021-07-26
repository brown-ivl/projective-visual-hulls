from VisualCone import VisualCone
import glob
import numpy as np
from PIL import Image, ImageFilter
import linecache
import matplotlib.pyplot as plt
import re

def get_silhouette(img_file):
    return Image.open(img_file) \
        .convert('1') \
        .filter(ImageFilter.BLUR) \
        .filter(ImageFilter.MinFilter(3)) \
        .filter(ImageFilter.MinFilter)


def get_parameter_string(parameter_file, silhouette_num):
    return linecache.getline(parameter_file, silhouette_num + 2).split()[1:]


def get_intrinsic_matrix(parameter_str):
    return np.array([[float(parameter_str[0]), float(parameter_str[1]), float(parameter_str[2])],
                     [float(parameter_str[3]), float(parameter_str[4]), float(parameter_str[5])],
                     [float(parameter_str[6]), float(parameter_str[7]), float(parameter_str[8])]])


def get_extrinsic_matrix(parameter_str):
    R = np.array([[float(parameter_str[9]), float(parameter_str[10]), float(parameter_str[11])],
                  [float(parameter_str[12]), float(parameter_str[13]), float(parameter_str[14])],
                  [float(parameter_str[15]), float(parameter_str[16]), float(parameter_str[17])]])
    t = np.array([[float(parameter_str[18])], [float(parameter_str[19])], [float(parameter_str[20])]])

    return np.hstack((R, t))


def plot_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    print("Plotting points...")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    print("Displaying plot!")
    plt.show()


parameter_file = 'data/temple_par.txt'
angle_file = 'data/temple_ang.txt'


def create_cone(img_file):
    img = get_silhouette(img_file)
    img = img.resize((img.width//10, img.height//10))
    img = np.asarray(img, dtype='float32')

    num = int(re.search(r'\d+', img_file).group()) - 1

    parameter_str = get_parameter_string(parameter_file, num)
    intrinsic_matrix = get_intrinsic_matrix(parameter_str)
    extrinsic_matrix = get_extrinsic_matrix(parameter_str)

    return VisualCone(intrinsic_matrix, extrinsic_matrix, img)


if __name__ == "__main__":

    #    print(intersection(*[Line3D((-2,-1.0,0.0), ()), Line3D((),()]))
    print("Creating cones...")
    cones = map(create_cone, sorted(glob.glob('data/*.png')))

    intersections = []
    print("Computing intersections...")
    for cone1 in cones:
        for cone2 in cones:
            if cone1 is not cone2:
                intersections += cone1.get_cone_intersection(cone2)
            print(intersections)
        print(intersections)