import os
import time
import numpy as np
import netCDF4 as nc
from FilterRoutinesNM import write_gmsh_mesh, write_2dm_mesh

import jigsawpy


# define experiment name
experiment = "NWA_test"

# define src_path and dst_path
src_path = f"/home/gsu000/projects/WW3-tools/unst_msh_gen/regional/NWA"
dataDir = "/home/gsu000/data/ppp7/RDWPS"
dst_path = os.path.join(dataDir, f'{experiment}_jigsawpy')
if not os.path.exists(dst_path): os.makedirs(dst_path)

jigsaw_mesh = True # set to False if already created

if jigsaw_mesh:

    TopographyFile = "/home/gsu000/projects/WW3-tools/unst_msh_gen/RTopo_2_0_4_GEBCO_v2023_60sec_pixel.nc"
    
    # directory to write output files to
    # tanh spacing
    hbeta = 0.01 # tanh width # make negative to have uniform spacing
    hcenter = 50
    hmin=1.0
    hmax=5.0 # test uniform grid
    dhdx=0.2
    
    
    opts = jigsawpy.jigsaw_jig_t()
    
    topo = jigsawpy.jigsaw_msh_t()
    
    geom = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()
    hmat = jigsawpy.jigsaw_msh_t()
    
    proj = jigsawpy.jigsaw_prj_t()
    
    ##--------------------------------- setup files for JIGSAW
    
    opts.geom_file = \
        os.path.join(dst_path, "geom.msh")
    
    opts.jcfg_file = \
        os.path.join(dst_path, "NWA.jig")
    
    opts.mesh_file = \
        os.path.join(dst_path, "mesh.msh")
    
    opts.hfun_file = \
        os.path.join(dst_path, "spac.msh")
    
    ##--------------------------------- define JIGSAW geometry
    
    jigsawpy.loadmsh(os.path.join(
        src_path, "NWA_geom_nosimplify.msh"), geom)
    
    data = nc.Dataset(TopographyFile,"r")
    xlon = np.asarray(data["lon"][:])
    ylat = np.asarray(data["lat"][:])
    elev = np.asarray(data["bed_elevation"][:]) + \
           np.asarray(data["ice_thickness"][:])
    xmid = 0.5*(xlon[:-1] + xlon[1:]) #* np.pi / 180.
    ymid = 0.5*(ylat[:-1] + ylat[1:]) #* np.pi / 180.
    
    xmin = np.min(
        geom.point["coord"][:, 0])
    ymin = np.min(
        geom.point["coord"][:, 1])
    xmax = np.max(
        geom.point["coord"][:, 0])
    ymax = np.max(
        geom.point["coord"][:, 1])
    
    zlev = elev
    
    xmsk = np.logical_and(xmid > xmin,
                          xmid < xmax)
    ymsk = np.logical_and(ymid > ymin,
                          ymid < ymax)
    
    zlev = zlev[:, xmsk]
    zlev = zlev[ymsk, :]
    
    ##--------------------------------- define spacing pattern
    
    hmat.mshID = "ellipsoid-grid"
    hmat.radii = np.full(
        +3, +6371.0,
        dtype=jigsawpy.jigsaw_msh_t.REALS_t)
    
    hmat.xgrid = \
        xmid[xmsk] * np.pi / 180.
    hmat.ygrid = \
        ymid[ymsk] * np.pi / 180.
    
    hmin = +1.0E+01; hmax = +1.0E+02
    
    hmat.value = \
        np.sqrt(np.maximum(-zlev, 0.)) / 0.5
    
    hmat.value = \
        np.maximum(hmat.value, hmin)
    hmat.value = \
        np.minimum(hmat.value, hmax)
    
    hmat.slope = np.full(
        hmat.value.shape, +0.1500,
        dtype=jigsawpy.jigsaw_msh_t.REALS_t)
    
    ##--------------------------------- do stereographic proj.
    
    geom.point["coord"][:, :] *= np.pi / 180.
    
    proj.prjID = 'stereographic'
    proj.radii = +6.371E+003
    proj.xbase = \
        +0.500 * (xmin + xmax) * np.pi / 180.
    proj.ybase = \
        +0.500 * (ymin + ymax) * np.pi / 180.
    
    jigsawpy.project(geom, proj, "fwd")
    jigsawpy.project(hmat, proj, "fwd")
    
    jigsawpy.savemsh(opts.geom_file, geom)
    jigsawpy.savemsh(opts.hfun_file, hmat)
    
    ##--------------------------------- set HFUN grad.-limiter
    
    jigsawpy.cmd.marche(opts, hmat)
    
    ##--------------------------------- make mesh using JIGSAW
    
    opts.hfun_scal = "absolute"
    opts.hfun_hmax = float("inf")       # null HFUN limits
    opts.hfun_hmin = float(+0.00)
    
    opts.mesh_dims = +2                 # 2-dim. simplexes
    opts.mesh_eps1 = +1.
    
    ttic = time.time()
    
    jigsawpy.cmd.jigsaw(opts, mesh)
    
    ttoc = time.time()
    
    print("CPUSEC =", (ttoc - ttic))
    
    cost = jigsawpy.triscr2(            # quality metrics!
        mesh.point["coord"],
        mesh.tria3["index"])
    
    print("TRISCR =", np.min(cost), np.mean(cost))
    
    cost = jigsawpy.pwrscr2(
        mesh.point["coord"],
        mesh.power,
        mesh.tria3["index"])
    
    print("PWRSCR =", np.min(cost), np.mean(cost))
    
    tbad = jigsawpy.centre2(
        mesh.point["coord"],
        mesh.power,
        mesh.tria3["index"])
    
    print("OBTUSE =",
          +np.count_nonzero(np.logical_not(tbad)))
    
    # need to convert for ww3 gmsh writer
    # transform mesh nodes to degree lat, lon
    # project mesh nodes to radian lat, lon
    jigsawpy.project(mesh, proj, "inv") # This used to work
    mesh.point["coord"] = mesh.point["coord"]*180. / np.pi
        
    # save mesh in degree lat, lon system
    jigsawpy.savemsh(os.path.join(dst_path,"NWA.LL.msh"),mesh)
        
    # create jigsaw R3 mesh on global surface and save to 
    S2=mesh.point["coord"][:,[0,1]]
    S2=S2*np.pi/180.
        
    R3=jigsawpy.S2toR3(mesh.radii,S2)
    
    meshR3 = jigsawpy.jigsaw_msh_t()
    mesh.mshID = 'ellipsoid-mesh'
    meshR3.tria3=mesh.tria3
    meshR3.ndims=3
    #make 3D coordinates 
    nd=R3.shape
    meshR3.vert3 = np.zeros(nd[0], dtype=mesh.VERT3_t)
    meshR3.vert3["coord"] = R3
    jigsawpy.savemsh(os.path.join(dst_path,'NWA.R3.msh'), meshR3)

from FilterRoutinesNM import *

#replace mesh with R3 mesh, mesh->mesh R3
jigsawpy.loadmsh(os.path.join(dst_path,'NWA.R3.msh'), mesh) # uncomment if starting here

opts.geom_file = "geom.msh"  #saves the geometry info for jigsaw
opts.jcfg_file = "opts.jig"  #jigsaw ctlr file


geom.mshID = "ellipsoid-mesh"
geom.radii = np.full(3, 6.371E+003, dtype=geom.REALS_t)

jigsawpy.savemsh(opts.geom_file, geom)

inject_dem()
filter_ocn()

jigsawpy.savemsh(os.path.join(dst_path,"NWA.F.R3.msh"), mesh)

# viz. in eg. paraview
#jigsawpy.savevtk(dst_path+"test.vtk", mesh)

# convert to lon lat
point = mesh.point["coord"]
point = jigsawpy.R3toS2(geom.radii, point)
point*= 180. / np.pi

depth = np.reshape(-1*mesh.value, (mesh.value.size, 1))
depth[depth <= 0] = 2
point = np.hstack((point, depth))  # append elev. as 3rd coord.
cells = [("triangle", mesh.tria3["index"])]
tri_data=cells[0][1]+1

#put coordinates in non standard format to avoid international date line
lon=point[:,0]
lon[np.where(lon>90)]=lon[np.where(lon>90)]-360
point[:,0]=lon

write_gmsh_mesh(os.path.join(dst_path, "NWA_test.ww3"), point, tri_data)
write_2dm_mesh(os.path.join(dst_path, "NWA_test.2dm"), point, tri_data)

#    jigsawpy.savevtk(os.path.join(
#        dst_path, "NWA_test.vtk"), mesh)
#-------------------------------- saving as ww3 file
#    # convert to lon lat    
#    jigsawpy.project(mesh, proj, "inv")
#
#    mesh = inject_dem(mesh, zlev, xmid[xmsk], ymid[ymsk])
#
#    point = mesh.point["coord"]
#    #point = jigsawpy.R3toS2(geom.radii, point) 
#    point*= 180. / np.pi
#     
#    depth = np.reshape(-1*mesh.value, (mesh.value.size, 1))
#    depth[depth <= 0] = 2
#    point = np.hstack((point, depth))  # append elev. as 3rd coord.
#    cells = [("triangle", mesh.tria3["index"])]
#    tri_data=cells[0][1]+1
#    
#    #put coordinates in non standard format to avoid international date line
#    lon=point[:,0]
#    lon[np.where(lon>90)]=lon[np.where(lon>90)]-360
#    point[:,0]=lon
#    
#    write_gmsh_mesh(os.path.join(dst_path,"NWA_test.ww3"), point, tri_data)
#------------------------------------ save mesh for Paraview

#    print("Saving to ../cache/case_6a.vtk")
#
#    jigsawpy.savevtk(os.path.join(
#        dst_path, "case_6a.vtk"), mesh)
#
#    print("Saving to ../cache/case_6b.vtk")
#
#    jigsawpy.savevtk(os.path.join(
#        dst_path, "case_6b.vtk"), hmat)

