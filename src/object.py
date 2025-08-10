import numpy as np
import time

from scipy.spatial import KDTree

import pyvista as pv


pcd_pv = pv.read('/home/sahana/Documents/github/object_detection_point_cloud_data/MLS_UTWENTE_super_sample.ply')
#print(pcd_pv)
pcd_pv.plot(eye_dome_lighting=True)


pcd_pv['elevation'] = pcd_pv.points[:,2]
pcd_pv['random'] = pcd_pv.points[:,0] * pcd_pv.points[:,1]
pv.plot(pcd_pv, scalars = pcd_pv['random'], 
        render_points_as_spheres=True, point_size=5,
        show_scalar_bar=False)


temp = pcd_pv.find_closest_point((1,1,0), n = 20)
print(temp)

tree = KDTree(pcd_pv.points)
t0 = time.time()
dists, indices = tree.query(pcd_pv.points, k = 20)

neighbors = pcd_pv.points[indices]
print(len(neighbors))
t1 = time.time()
print(f"Neighbor Computation in {t1-t0} seconds")