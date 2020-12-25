import gmsh

# using gmsh
def gmsh_op(filenamein,filenameout,dogui):
    # initialize the gmsh
    gmsh.initialize()
    
    # open msh mesh file
    gmsh.open(filenamein)
    
    # try to smooth the mesh 
    gmsh.option.setNumber("Mesh.Smoothing", 10000) # 10000 steps, maybe too large
    gmsh.option.setNumber("Mesh.SmoothRatio",5)

    # # another smoother with entities
    # for s in gmsh.model.getEntities(2):
    #     gmsh.model.mesh.setSmoothing(s[0], s[1], 100)

    # generate a 2d mesh
    gmsh.model.mesh.generate(2)
    # write the smoothed msh file
    gmsh.write(filenameout)

    # launch the GNI to see the results:
    if dogui:
        gmsh.fltk.run()

    # finalize the gmsh lib
    gmsh.finalize()
