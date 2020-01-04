import numpy as np
from scipy.spatial import cKDTree as KDTree


def chamfer_distance(points1, points2):
    """This function computes a symmetric chamfer distance, i.e. the sum of both directions.
    The point distance is defined as squared distance.

    :param: points1: numpy array for K-dimension points. (N1, K)
    :param: points2: numpy array for K-dimension points. (N2, K)
    """

    # one direction
    points2_kd_tree = KDTree(points2)
    one_distances, one_vertex_ids = points2_kd_tree.query(points1)
    A_to_B_chamfer = np.mean(np.square(one_distances))

    # other direction
    points1_kd_tree = KDTree(points1)
    two_distances, two_vertex_ids = points1_kd_tree.query(points2)
    B_to_A_chamfer = np.mean(np.square(two_distances))

    cd = A_to_B_chamfer + B_to_A_chamfer
    return cd


if __name__ == '__main__':
	pts1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
	pts2 = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])

	cd = chamfer_distance(pts1, pts2) # should be 2.0
	print(cd)
