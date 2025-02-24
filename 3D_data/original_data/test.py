import h5py
import os
import numpy as np
from IPython.display import Image
from voxelgrid import VoxelGrid
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io

if __name__ == '__main__':
    with h5py.File("./train_point_clouds.h5", "r") as hf:
        a = hf["0"]

        digit_a = (a["img"][:], a["points"][:], a.attrs["label"])

    plt.title("DIGIT A: " + str(digit_a[2]))
    plt.imshow(digit_a[0])

    a_voxelgrid = VoxelGrid(digit_a[1], x_y_z=[25, 25, 25])
    a_voxelgrid.build()
    voxelgrid = a_voxelgrid.vector
    scipy.io.savemat('voxelgrid_sample0.mat', {'voxelgrid': voxelgrid})
    # num_cubes = 100
    # indices = np.random.randint(0, 16, size=(num_cubes, 3))
    # voxelgrid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

    # 创建一个三维图形对象
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # 遍历voxelgrid，绘制立方体
    for x in range(24):
        for y in range(24):
            for z in range(24):
                if voxelgrid[x, y, z] > 1:
                    ax.scatter(x, y, z, color='b', marker='s')

    # 设置坐标轴范围
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 25)
    ax.set_zlim(0, 25)

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.view_init(elev=90, azim=0)
    # 显示图形
    plt.show()
    fig.savefig("img0.png")


