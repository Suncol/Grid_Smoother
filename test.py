import os
import numpy as np
import h5py

import smooth_core.core_smoother as core_smoother
import smooth_core.data_io as data_io
import smooth_core.ploter as ploter 

# -----------------------input-------------------------
# x, y = data_io.read_grid_file('lfm_grid_65x65.mat', 'x', 'y')
f = h5py.File('LFM_dayside_40.h5','r')
x = np.transpose(f['/xe'])
y = np.transpose(f['/ye'])

ploter.plotGrid(x, y,'Before Smoothing')

#x, y = core_smoother.elliptic_mesh(x, y, 0.01, 1000)
#x, y = core_smoother.edge_fit(x, y)

# --------------deal with the boundary and TFI--------

u, v = core_smoother.BoundaryNormalization(x, y)

xi, eta = core_smoother.BoundaryBlendedControlFunction(u,v)

x, y = core_smoother.TFI(x,y,xi,eta)

# ----------------laplace smoothing-------------------

x, y, residual = core_smoother.laplacesmoothing(x,y,0.2,1.0e-4,100) # omega=0.2 err=1.0e-4 maxit=1000
ploter.plotGrid(x, y,'After LaplaceSmoothing')

#data_io.write_data('D:\\Program\\MHD\\Grid\\gamera_grid_smooth.mat',x,y,'x_s','y_s')

## do stupid things
iMax, jMax = x.shape

x = np.hstack((x[:,:jMax//2-1], \
    np.reshape((x[:,jMax//2-1]+x[:,jMax//2+1])/2.0,(iMax,1)), \
         x[:,jMax//2+1:]))

y = np.hstack((y[:,:jMax//2-1], \
    np.reshape((y[:,jMax//2-1]+y[:,jMax//2+1])/2.0, (iMax,1)),\
         y[:,jMax//2+1:]))

y[:,jMax//2-2] = (y[:,jMax//2-2]-y[:,jMax//2])/1.7
y[:,jMax//2] = -1 * y[:,jMax//2-2]

y[:,jMax//2-1] = 0.0 

# -------------------elliptic smoothing----------------

x, y = core_smoother.elliptic_mesh(x, y, 0.01, 1000)
ploter.plotGrid(x, y, 'After Elliptic Smoothing') # ok for now 


# ---------------output smoothed mesh grid -------

# slice the mesh grid to half plane
x_s = x[:,1:jMax//2]
y_s = y[:,1:jMax//2]
ploter.plotGrid(x_s, y_s, 'Half Plane Mesh Grid')

# save the grid result to mat file, for now we use the result here
# you can use the msh lib and have a try
dirName = 'results/'

if not os.path.exists(dirName):
    os.mkdir(dirName)
else:
    pass


data_io.write_grid_file(os.path.join( \
    dirName,'gamera_grid_smooth.mat'), \
        x_s,y_s,'x_s','y_s')

