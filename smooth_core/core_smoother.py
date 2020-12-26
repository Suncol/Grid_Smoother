import math
import numpy as np
from scipy.optimize import curve_fit
from numba import jit # using numba jit to speed up

import smooth_core.ploter as ploter

# A 2d mesh smoother by solving the elliptic mesh pde. 

@jit(nopython=True)
def elliptic_mesh(x, y, convCrit, maxit): 

    '''
    Input:
    conCrit: convergence criteria
    maxit: max allowed number of iters

    '''

    # combine the x and y mesh grid to a complex array
    xy = np.vectorize(complex)(x, y)    

    # get the shape    
    m, n = xy.shape

    px = (1,n-1)
    e = (2,n) # east side of the mesh grid
    w = (0,n-2) # west side of the mesh grid

    py = (1,m-1)
    n = (2,m)
    s = (0,m-2)

    for i in range(maxit):

        # set mesh grid location
        xyE = xy[py[0]:py[1],e[0]:e[1]]
        xyN = xy[n[0]:n[1], px[0]:px[1]]
        xyW = xy[py[0]:py[1],w[0]:w[1]]
        xyS = xy[s[0]:s[1],px[0]:px[1]]
        
        xyNE = xy[n[0]:n[1],e[0]:e[1]]
        xySE = xy[s[0]:s[1],e[0]:e[1]]
        xyNW = xy[n[0]:n[1],w[0]:w[1]]
        xySW = xy[s[0]:s[1],w[0]:w[1]]
      
        # calculate derivatives
        xy_xi = (xyE - xyW)/2.0
        xy_et = (xyN - xyS)/2.0
        xy_xiet = (xyNE - xySE - xyNW + xySW)/4.0

        # calcualte coefficients
        a = np.absolute(xy_et) * np.absolute(xy_et) 
        b = np.real(xy_xi) * np.real(xy_et) + np.imag(xy_xi) * np.imag(xy_et) 
        c = np.absolute(xy_xi) * np.absolute(xy_xi)
        
        # solve for the new interior mesh verts
        xyP = (a * (xyE + xyW) + c * (xyN + xyS) - 2.0 * b *(xy_xiet)) / (2.0*(a + c))
        
        # check for convergence 
        err = np.max(np.absolute(xyP - xy[py[0]:py[1],px[0]:px[1]]))
        if (err < convCrit):
            print('-----mesh converged in '+ str(i) + ' interations \n')
            print('-----err: '+ str(err)+' --------- \n')
            break

        xy[py[0]:py[1], px[0]:px[1]] = xyP # update the mesh verts, but the edge points are not update!!!
    
    xyout = xy # no need but for read friendly

    x = np.real(xyout)
    y = np.imag(xyout)
    return x, y 


# laplace smoothing
@jit(nopython=True)
def laplacesmoothing(x, y, omega, targetError, maxit):
    iMax, jMax = x.shape

    Rx = np.zeros(shape=(iMax-1, jMax-1))
    Ry = np.zeros(shape=(iMax-1, jMax-1))

    iteration = 0
    error = 1
    lastRValue = 1
    residual = []

    while (error > targetError):
        for i in range(1,iMax-1):
            for j in range(1,jMax-1):
                xXi = (x[i+1,j]-x[i-1,j])/2
                yXi = (y[i+1,j]-y[i-1,j])/2
                xEta = (x[i,j+1]-x[i,j-1])/2
                yEta = (y[i,j+1]-y[i,j-1])/2
                J = xXi*yEta - xEta*yXi

                alpha = xEta**2 + yEta**2
                beta = xXi*xEta + yXi*yEta
                gamma = xXi**2 + yXi**2

                # Finding X
                Rx1 = alpha*(x[i+1,j] - 2*x[i,j] + x[i-1,j])
                Rx2 = (-0.5)*beta*(x[i+1,j+1] - x[i-1,j+1] - x[i+1,j-1] + x[i-1,j-1])
                Rx3 = gamma*(x[i,j+1] - 2*x[i,j] + x[i,j-1])
                Rx[i,j] = Rx1 + Rx2 + Rx3

                # Finding Y
                Ry1 = alpha*(y[i+1,j] - 2*y[i,j] + y[i-1,j])
                Ry2 = (-0.5)*beta*(y[i+1,j+1] - y[i-1,j+1] - y[i+1,j-1] + y[i-1,j-1])
                Ry3 = gamma*(y[i,j+1] - 2*y[i,j] + y[i,j-1])
                Ry[i,j] = Ry1 + Ry2 + Ry3

                # Update X and Y
                x[i,j] = x[i,j] + omega*((Rx[i,j])/(2*(alpha + gamma)))
                y[i,j] = y[i,j] + omega*((Ry[i,j])/(2*(alpha + gamma)))
        
        # Find residual
        currentRValue = np.sqrt(np.sum(Rx)**2 + np.sum(Ry)**2)
        error = abs(lastRValue - currentRValue)
        
        # Store residual
        iteration = iteration + 1
        
        # Other escape routes
        if (iteration>maxit):
            break
        print('-------laplace smoothing iteration: ' + str(iteration) + '------------' )
        residual.append(error*100) # per cent
        
        # Update value
        lastRValue = currentRValue

    return (x, y, residual)


# fit the edge points that make the x axis bad, the most outside point?
@jit(nopython=True)
def edge_fit(x, y):
    iMax, jMax = x.shape
    for i in range(1,iMax-1):
        coef = np.polyfit(x[i,1:-1],y[i,1:-1],11) 

        # polyfunc = lambda x : coef[0]*np.power(x,3) + \ # i think poly5 is ok
        #     coef[1]*np.power(x,2) + coef[2]*np.power(x,1) + \
        #     coef[3]        
        polyfunc = np.poly1d(coef)
        x[i,-1] = biSection(x[i+1,-1],x[i-1,-1],1,polyfunc,10000)
        print('-----------edge fitting------')
        print('OK for i = ', i )
    
    print('-----------end edge fitting---------')
    return (x, y)


# get the func root
@jit(nopython=True)
def biSection(a,b,threshold,f,maxit):
    iter=0
    while a<b:
        mid = a + abs(b-a)/2.0
        if abs(f(mid)) < threshold:
            return mid
        if f(mid)*f(b) < 0:
            a = mid
        if f(a)*f(mid) < 0:
            b=mid
        iter+=1

        if iter > maxit:
            break

    return (a + b) / 2.0

@jit(nopython=True)
def BoundaryNormalization(X, Y):
    iMax, jMax = X.shape 

    # Normalization at boundary
    meshLength = np.zeros(shape=(iMax-1, jMax-1))
    u = np.zeros(shape=(iMax, jMax))
    v = np.zeros(shape=(iMax, jMax))

    # Bottom
    totalLength = 0
    for i in range(iMax-1):
        dx = X[i+1,0] - X[i,0]
        dy = Y[i+1,0] - Y[i,0]
        dLength = math.sqrt(dx**2 + dy**2)
        totalLength = totalLength + dLength
        meshLength[i,0] = totalLength

    for i in range(iMax-1):
        u[i+1,0] = meshLength[i,0]/totalLength
        v[i+1,0] = 0

    # Top
    totalLength = 0
    for i in range(iMax-1):
        dx = X[i+1,jMax-1] - X[i,jMax-1]
        dy = Y[i+1,jMax-1] - Y[i,jMax-1]
        dLength = math.sqrt(dx**2 + dy**2)
        totalLength = totalLength + dLength
        meshLength[i,jMax-2] = totalLength
        
    for i in range(iMax-1):
        u[i+1,jMax-1] = meshLength[i,jMax-2]/totalLength
        v[i+1,jMax-1] = 1

    # reset
    meshLength = np.zeros(shape=(iMax-1, jMax-1))

    # Left
    totalLength = 0
    for i in range(jMax-1):
        dx = X[0,i+1] - X[0,i]
        dy = Y[0,i+1] - Y[0,i]
        dLength = math.sqrt(dx**2 + dy**2)
        totalLength = totalLength + dLength
        meshLength[0,i] = totalLength

    for i in range(jMax-1):
        u[0,i+1] = 0
        v[0,i+1] = meshLength[0,i]/totalLength
        
    # Right
    totalLength = 0
    for i in range(jMax-1):
        dx = X[iMax-1,i+1] - X[iMax-1,i]
        dy = Y[iMax-1,i+1] - Y[iMax-1,i]
        dLength = math.sqrt(dx**2 + dy**2)
        totalLength = totalLength + dLength
        meshLength[iMax-2,i] = totalLength

    for i in range(jMax-1):
        u[iMax-1,i+1] = 1
        v[iMax-1,i+1] = meshLength[iMax-2,i]/totalLength

    return (u, v)

@jit(nopython=True)
def BoundaryBlendedControlFunction(u, v):
    iMax, jMax = u.shape

    # Boundary-Blended Control Functions
    for i in range(iMax-1):
        for j in range(jMax-1):
            part1 = (1-v[0,j])*u[i,0] + v[0,j]*u[i,jMax-1]
            part2 = 1 - (u[i,jMax-1]-u[i,0])*(v[iMax-1,j]-v[0,j])
            u[i,j] = part1/part2

            part1 = (1-u[i,0])*v[0,j] + u[i,0]*v[iMax-1,j]
            part2 = 1 - (v[iMax-1,j]-v[0,j])*(u[i,jMax-1]-u[i,0])
            v[i,j] = part1/part2

    return (u, v)

@jit(nopython=True)
def TFI(X, Y, u, v):    
    iMax, jMax = X.shape

    # Transfinite Interpolation
    for i in range(1,iMax-1):
        for j in range(1,jMax-1):
            U = (1-u[i,j])*X[0,j] + u[i,j]*X[iMax-1,j]
            V = (1-v[i,j])*X[i,0] + v[i,j]*X[i,jMax-1]
            UV = u[i,j]*v[i,j]*X[iMax-1,jMax-1] + u[i,j]*(1-v[i,j])*X[iMax-1,0] +\
                (1-u[i,j])*v[i,j]*X[0,jMax-1] + (1-u[i,j])*(1-v[i,j])*X[0,0]
            X[i,j] = U + V - UV

            U = (1-u[i,j])*Y[0,j] + u[i,j]*Y[iMax-1,j]
            V = (1-v[i,j])*Y[i,0] + v[i,j]*Y[i,jMax-1]
            UV = u[i,j]*v[i,j]*Y[iMax-1,jMax-1] + u[i,j]*(1-v[i,j])*Y[iMax-1,0] +\
                (1-u[i,j])*v[i,j]*Y[0,jMax-1] + (1-u[i,j])*(1-v[i,j])*Y[0,0]
            Y[i,j] = U + V - UV

    return (X, Y)