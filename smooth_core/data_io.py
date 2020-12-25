import gmsh # python gmsh, default we load the gui interface
import pyvista as pv # python mesh tools
from tvtk.api import tvtk,write_data
import meshio # for mesh file io, you can just use the command line tools of it
import numpy as np
import scipy.io # for io mat file with matlab

# mat file reader
# usually we input the x,y as the meshgrid
def read_grid_file(filename,x_name,y_name):
    gdata = scipy.io.loadmat(filename)
    x = gdata[x_name]
    y = gdata[y_name]
    return x, y # x, y meshgrid

# mat file writer
def write_grid_file(filename,x,y,x_name,y_name):
    scipy.io.savemat(filename,{x_name:x,y_name:y})

# save mesh file 
def save_mesh(filename,ggrid): 
    pv.save_meshio(filename,ggrid,binary=False) # for reading friendly, save to text


# save the vtk file
def save_vtk_file(x,y,filenameout): # can be view in paraview
    x_ravel = x.ravel()
    y_ravel = y.ravel()
    z_ravel = np.zeros_like(x_ravel)
    
    points = np.stack((x_ravel,y_ravel,z_ravel),axis=1)

    quads = []

    for x_index in range(x.shape[0]-1): # now get all quads in grids
        for y_index in range(y.shape[1]-1):
            q = [x_index+1, x_index , (y_index+1)*x.shape[0]+x_index, (y_index+1)*x.shape[0]+x_index+1] # Note here that I am using CW ordering not CCW
            quads.append(q)

    poly_edge = np.array(quads)

    mesh = tvtk.PolyData(points=points, polys=poly_edge)

    write_data(mesh, filenameout)

# easy create 2D structured mesh grid
def create_grid(x,y):
    z = np.zeros_like(x)
    ggrid = pv.StructuredGrid(x,y,z)
    return ggrid



# for unit test 
if __name__ == "__main__":

    pass

    # # test if read mat file is ok
    # x, y = read_grid_file("lfm_grid_65x65_backup.mat",'x','y')
    # print(x.shape,y.shape)

    # # test create grid
    # ggrid = create_grid(x,y)

    # # test grid save to msh file for gmsh 
    # save_mesh("gamera_grid.msh",ggrid)

    



