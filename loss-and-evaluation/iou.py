import numpy as np


def IoUarr(arr1, arr2, inside=0, reduce_mean=True):
    """Calculate IoU for two batched input binary arrays. Default 0 for inside, 1 for outside.

    :param arr1: ndarray. (B, D1, D2, ...)
    :param arr2: ndarray. (B, D1, D2, ...)
    :param inside: int. Inside label, 0 or 1.
    :param reduce_mean: Boolean. Reduce mean over batch.
    :return:
    """
    if not arr1.shape == arr2.shape:
        raise ValueError("Two input arrays should be of equal size.")

    if not inside == 1:
        arr1 = (1 - arr1).astype(np.bool)
        arr2 = (1 - arr2).astype(np.bool)
    
    axes = tuple(range(1, len(arr1.shape)))
    intersection = np.sum(np.logical_and(arr1, arr2), axes)
    union = np.sum(np.logical_or(arr1, arr2), axes)
    iou = intersection / union

    if reduce_mean:
        iou = np.mean(iou)
    return iou


def IoUaabb(bbox1, bbox2, reduce_mean=True):
    """Calculate IoU for two batched axis-aligned bounding box. Support both 2D and 3D box.

    :param bbox1: ndarray. 3D box: (B, 6), 6 for (min_x, min_y, min_z, max_x, max_y, max_z);
                           2D box: (B, 4), 4 for (min_x, min_y, max_x, max_y).
    :param bbox2: ndarray. Same as bbox1
    :return:
    """
    volume1 = bboxVolume(bbox1)
    volume2 = bboxVolume(bbox2)
    
    min_b1, max_b1 = np.split(bbox1, 2, axis=1)
    min_b2, max_b2 = np.split(bbox2, 2, axis=1)
    min_inter = np.maximum(min_b1, min_b2)
    max_inter = np.minimum(max_b1, max_b2)
    box_inter = np.concatenate([min_inter, max_inter], axis=1)
    
    volume_inter = bboxVolume(box_inter)
    volume_union = volume1 + volume2 - volume_inter
    iou = volume_inter / volume_union

    if reduce_mean:
        iou = np.mean(iou)
    return iou


def bboxVolume(bbox):
    """Calculate volume for batched bounding box. Support both 2D and 3D box.

    :param bbox: ndarray. 3D box: (B, 6), 6 for (min_x, min_y, min_z, max_x, max_y, max_z);
                          2D box: (B, 4), 4 for (min_x, min_y, max_x, max_y).
    :return:
    """
    d = bbox.shape[1] // 2
    size = bbox[:, d:] - bbox[:, :d]
    volume = np.product(size, axis=1)
    return volume



if __name__ == '__main__':
    # test IoUarr
    img1 = np.ones((1, 8, 8))
    img1[0, 3:5, 3:5] = 0

    img2 = np.ones((1, 8, 8))
    img2[0, 4:6, 4:6] = 0
    iou = IoUarr(img1, img2) # 1 / 7
    print(iou)

    # test IoUaabb
    bbox1 = np.array([0, 0, 2, 2]).reshape((1, -1))
    bbox2 = np.array([-1, -1, 1, 1]).reshape((1, -1))
    iou = IoUaabb(bbox1, bbox2) # 1 / 7
    print(iou)

    bbox1 = np.array([0, 0, 0, 2, 2, 2]).reshape((1, -1))
    bbox2 = np.array([0, 0, 0, 1, 1, 1]).reshape((1, -1))
    iou = IoUaabb(bbox1, bbox2) # 1 / 4
    print(iou)
