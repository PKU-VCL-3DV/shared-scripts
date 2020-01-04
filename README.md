# shared-scripts
Commonly used scripts to share.

## Visualization
[visualize.py](https://github.com/PKU-VCL-3DV/shared-scripts/blob/master/visualize.py) can be used to quickly visualize mesh, point cloud, voxel, part box assembly and save rendered image if specified. Both one object or a directory or objects is supported.  
```
python visualize.py --format {file format} --src {object path or directory to visualize} --save_image[optional]
```
Mesh is visualized by [trimesh](https://github.com/mikedh/trimesh), point cloud by [open3d](http://www.open3d.org) and voxel/box by matplotlib.

## Loss and Evaluation
[loss-and-evaluation](https://github.com/PKU-VCL-3DV/shared-scripts/tree/master/loss-and-evaluation) provides code for some distance metrics.
### IoU
- [iou.py](https://github.com/PKU-VCL-3DV/shared-scripts/tree/master/loss-and-evaluation/iou.py) provides functions to calculate IoU for binary arrays or axis-aligned bounding boxes. Implemented in numpy but should be easy converted to pytorch or tensorflow.
### Chamfer Distance
- [chamfer.py](https://github.com/PKU-VCL-3DV/shared-scripts/tree/master/loss-and-evaluation/chamfer.py) provides functions to calculate symmetric chamfer distance, implemented in numpy. 
- https://github.com/chrdiller/pyTorchChamferDistance provides fast implementation, wrapped in pytorch.
### EMD Distance
- https://github.com/daerduoCarey/PyTorchEMD provides fast implementationm, wrapped in pytorch. (But cannot parralleled over multiple GPUs..) 
- https://github.com/optas/latent_3d_points/tree/master/external/structural_losses has fast tensorflow implementation. Chamfer distance is also included.
### Hausdorff Distance
- [directed_hausdorff.py](https://github.com/PKU-VCL-3DV/shared-scripts/tree/master/loss-and-evaluation/directed_hausdorff.py) provides pytorch implementation for directed hausdorff distance.
- Tensorflow implementation can be found [here](https://github.com/xuelin-chen/pcl2pcl-gan-pub/blob/master/pc2pc/structural_losses_utils/tf_hausdorff_distance.py).
### Light Field Distance
- pending...
