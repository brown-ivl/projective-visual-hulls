import matplotlib.pyplot as plt
from tk3dv.nocstools import datastructures as ds 
import numpy as np
import cv2

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
        ax.scatter(cone.xyzs[:, 0], cone.xyzs[:, 1], cone.xyzs[:, 2], c='r', marker='o', s=4)

        print("Plotting camera...")
        ax.scatter(cone.camera_location[0], cone.camera_location[1], cone.camera_location[2], c='b', marker='o')

    if show_pc:
        scale = 1.0
        ax.scatter(scale*(point_clouds[:,0]-0.5), scale*(point_clouds[:,1]-0.5), scale*(point_clouds[:,2]-0.5), s=4)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_xlim([-2,2])
    ax.set_ylim([-2,2])
    ax.set_zlim([-2,2])
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
    nocs_map = cv2.imread(path, -1)
    nocs_map = nocs_map[:, :, :3]
    nocs_map = cv2.cvtColor(nocs_map, cv2.COLOR_BGR2RGB)
    return nocs_map

    