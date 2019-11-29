import argparse
import os
import numpy as np
import json
import h5py
import time
import mcubes as libmcubes


def vis_mesh(paths, save_image=False):
    """visualize mesh files of format 'ply' or 'obj'. """
    import trimesh
    camera = None
    for path in paths:
        file_format = path.split('.')[-1]
        if file_format == 'ply' or file_format == 'obj':
            print("vis mesh: {}".format(path))
            mesh = trimesh.load(path)
        else:
            print("skip file: {}".format(path))
            continue

        scene = trimesh.Scene(mesh, camera_transform=camera)
        sceneview = scene.show()
        if camera is None:
            camera = scene.camera_transform
        if save_image:
            save_path = os.path.join(path[:-len(file_format)] + '.png')
            png = scene.save_image(resolution=[512, 512],
                                   visible=True)
            with open(save_path, 'wb') as f:
                f.write(png)
        sceneview.close()


def vis_point_cloud(paths, save_image=False):
    """visualize point cloud files of format 'ply' or 'pcd' or 'npy'. """
    import open3d as o3d
    for path in paths:
        file_format = path.split('.')[-1]
        if file_format == 'ply' or file_format == 'pcd':
            print("vis point cloud: {}".format(path))
            pcd = o3d.io.read_point_cloud(path)
        elif file_format == 'npy':
            print("vis point cloud: {}".format(path))
            shape_pcs = np.load(path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(shape_pcs)
        else:
            print("skip file: {}".format(path))

        o3d.visualization.draw_geometries([pcd])


def vis_voxel(paths, save_image=False):
    """visualize voxel of format 'npy' """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    for path in paths:
        file_format = path.split('.')[-1]
        if file_format == 'npy':
            print("vis voxel: {}".format(path))
            voxel = np.load(path)
        else:
            print("skip file: {}".format(path))
            continue

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(voxel, facecolors='b', edgecolors='k')  # this can be slow
        # ax.view_init(elev=40, azim=40) # set initial view angle position
        # plt.axis('off')  # hide axis
        if save_image:
            save_path = os.path.join(path[:-4] + '.png')
            plt.savefig(save_path, transparent=True)
        plt.show()
        plt.close()


def minmax2points(minmax):
    minp = minmax[:3]
    maxp = minmax[3:]
    P = np.asarray([minp,
         [maxp[0], minp[1], minp[2]],
         [maxp[0], minp[1], maxp[2]],
         [minp[0], minp[1], maxp[2]],
         [minp[0], maxp[1], minp[2]],
         [maxp[0], maxp[1], minp[2]],
         maxp,
         [minp[0], maxp[1], maxp[2]],
         ])
    P = P[:, [2, 1, 0]]  # FIXME: switch axis
    return P


def points2verts(Z):
    verts = [[Z[0], Z[1], Z[2], Z[3]],
             [Z[4], Z[5], Z[6], Z[7]],
             [Z[0], Z[1], Z[5], Z[4]],
             [Z[2], Z[3], Z[7], Z[6]],
             [Z[1], Z[2], Z[6], Z[5]],
             [Z[4], Z[7], Z[3], Z[0]]]
    return verts


def draw_parts_bbox(bboxes, ax, limit=64, transparency=0.6):
    """draw AABB of part box assembly with transparent face"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    n_parts = bboxes.shape[0]
    colors = [[255, 105, 180],
              [30, 144, 255],
              [127, 255, 212],
              [255, 215, 0],
              [139, 69, 19],
              [205, 102, 29],
              [255, 231, 186],
              [238, 197, 145],
              [139, 35, 35]]
    colors = np.asarray(colors) / 255
    # points = np.round(points).astype(np.int)
    for idx in range(n_parts):
        bbox = bboxes[idx]
        points = minmax2points(bbox)
        verts = points2verts(points)

        pc = Poly3DCollection(verts, linewidths=0.5, edgecolors='k', alpha=transparency)
        pc.set_facecolor(colors[idx % len(colors)])
        ax.add_collection3d(pc)

        # size = (bbox[3:] - bbox[:3]).tolist()
        # for s, e in combinations(np.array(list(product(bbox[[2, 5]], bbox[[1, 4]], bbox[[0, 3]]))), 2):
        #     if np.sum(np.abs(s - e)) in size:
        #         ax.plot3D(*zip(s, e), color=colors[idx % len(colors)], alpha=transparency)

    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_zlim(0, limit)


def vis_box(paths, save_image=False):
    """visualize axis-aligned bounding box(AABB) for part assembly of format 'npy' 
       each npy of shape (K, 6), K the number of parts, 6 the coordinates of min/max points
       (min.x, min.y, min.z, max.x, max.y, max.z)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    for path in paths:
        file_format = path.split('.')[-1]
        if file_format != 'npy':
            print("skip file: {}".format(path))
            continue

        print("vis box: {}".format(path))

        box_minmax = np.load(path)
        fig = plt.figure()
        ax1 = fig.gca(projection='3d')
        draw_parts_bbox(box_minmax, ax1)
        ax1.view_init(elev=40, azim=-40) # set initial view angle position
        plt.axis('off')  # hide axis

        if save_image:
            save_path = os.path.join(path[:-4] + '-box.png')
            plt.savefig(save_path, transparent=True)
        plt.show()
        plt.close()


def visualize(args):
    if os.path.isdir(args.src):
        paths = sorted([os.path.join(args.src, x) for x in os.listdir(args.src)])
    else:
        paths = [args.src]

    if args.format == 'mesh':
        vis_mesh(paths, args.save_image)
    elif args.format == 'pc':
        vis_point_cloud(paths, args.save_image)
    elif args.format == 'voxel':
        vis_voxel(paths, args.save_image)
    elif args.format == 'box':
        vis_box(paths, args.save_image)
    else:
        raise NotImplementedError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', type=str, default='mesh', choices=['mesh', 'pc', 'voxel', 'box'], 
        required=False, help="data format")
    parser.add_argument('--src', type=str, required=True, help="data path or directory")
    parser.add_argument('--save_image', action='store_true', default=False, help="save rendered image or not")
    args = parser.parse_args()

    visualize(args)


if __name__ == '__main__':
    main()
