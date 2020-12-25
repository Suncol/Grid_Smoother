import numpy as np
import pyvista as pv

# easy create 2D structured mesh grid
def create_grid(x,y):
    z = np.zeros_like(x)
    ggrid = pv.StructuredGrid(x,y,z)
    return ggrid

# create triangle mesh data, maybe loss some points at the edge of the mesh
# only for test
def get_tri_mesh(x,y,filename):
    z = np.zeros_like(x)
    points = np.c_[x.reshape(-1),y.reshape(-1),z.reshape(-1)]
    # simply pass the numpy points to the PolyData constructor
    cloud = pv.PolyData(points)
    #cloud.plot(point_size=5) # plot the points clouds
    surf = cloud.delaunay_2d(alpha=1.0) # remove some 
    #surf.plot(show_edges=True) # plot the triangle mesh
    #pv.save_meshio("tri_grid.msh",surf,binary=False) # save tri mesh to msh file
    save_mesh(filename,surf) 