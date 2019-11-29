# shared-scripts
Commonly used scripts to share.

## Visualization
[visualize.py](https://github.com/PKU-VCL-3DV/shared-scripts/blob/master/visualize.py) can be used to quickly visualize mesh, point cloud, voxel, part box assembly and save rendered image if specified. Both one object or a directory or objects is supported.  
```
python visualize.py --format {file format} --src {object path or directory to visualize} --save_image[optional]
```
Mesh is visualized by [trimesh](https://github.com/mikedh/trimesh), point cloud by [open3d](http://www.open3d.org) and voxel/box by matplotlib.
