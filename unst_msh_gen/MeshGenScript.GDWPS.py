
import os
import argparse
import time
import numpy as np
import netCDF4 as nc

import jigsawpy
from spacing import *

from scipy.interpolate import RegularGridInterpolator

#-------------------Input Files----------------------------------------
dataDir = "/home/gsu000/data/ppp7/GDWPS"
#GSSH coastline 
#jigsaw .msh format Planer Straight Line Graph defining mesh outer boundary and coastline
#PSLGFile="NWcoastal.PSLG.msh"
#PSLGFile=os.path.join(dataDir, "test.msh")
#PSLGFile = "./GDWPS_geom_nosimplify.msh"
#simplify_ext = "simplified_12thdeg"
simplify_ext = "nosimplify"
PSLGFile = f"./GDWPS_geom_{simplify_ext}.msh"
#jigsaw gridded .msh format Distance to taget poings
#DistanceToCoastFile="DFun.NWcoastal.PSLG.msh"
#jigsaw gridded .msh format topography on same grid as distance
#TopographyFile="Topo.DFun.NWcoastal.PSLG.msh"
TopographyFile = "/home/gsu000/projects/WW3-tools/unst_msh_gen/RTopo_2_0_4_GEBCO_v2023_60sec_pixel.nc"

# directory to write output files to
# tanh spacing
hbeta = 0.02 # tanh width # make negative to have uniform spacing
hcenter = 150
hmin=30.0
hmax=3.0 # test uniform grid
dhdx=0.3

exp = "GDWPS_geom_30km_to_3km_h0150_hb02"
outDir=os.path.join(dataDir, f'{exp}_jigsawpy')
# Create the output directory------------------------------------------
if not os.path.exists(outDir): os.makedirs(outDir)

# flag to check if I need to calculate the mesh or just apply filters
calculate_mesh = False
meshFile = os.path.join(outDir, "GDWPS.R3.msh")
    
if calculate_mesh or not os.path.exists(meshFile):
    #-------------------Paramter Inputs------------------------------------
    #parameters for specifying resolution
    #d0=15000.
    #d1=300000.
    #beta=2000.
    #Smin=0.25
    #Smax=7.5
    
    # Setup jigsaw structures----------------------------------------------
    opts = jigsawpy.jigsaw_jig_t()
    spac = jigsawpy.jigsaw_msh_t()
    geom = jigsawpy.jigsaw_msh_t()
    mesh = jigsawpy.jigsaw_msh_t()
    hmat = jigsawpy.jigsaw_msh_t()
    proj = jigsawpy.jigsaw_prj_t()
    
    opts.geom_file = os.path.join(outDir,"geom.msh")
    opts.jcfg_file = os.path.join(outDir,"jcfg.jig")
    opts.mesh_file = os.path.join(outDir,"mesh.msh")
    opts.hfun_file = os.path.join(outDir,"spac.msh")
    # load input data------------------------------------------------------
#    jigsawpy.loadmsh(PSLGFile, geom)
#    geom.point["coord"][:, :] *= np.pi / 180.
    geom.mshID = "ellipsoid-mesh"
    geom.radii = np.full(3, 6.371E+003, dtype=geom.REALS_t)
    jigsawpy.savemsh(opts.geom_file, geom)
   
    # calculate spacing function
    data = nc.Dataset(TopographyFile,"r")
    xlon = np.asarray(data["lon"][:])
    ylat = np.asarray(data["lat"][:])
    elev = np.asarray(data["bed_elevation"][:]) + \
           np.asarray(data["ice_thickness"][:])
    xmid = 0.5*(xlon[:-1] + xlon[1:]) #* np.pi / 180.
    ymid = 0.5*(ylat[:-1] + ylat[1:]) #* np.pi / 180.
    
    # get land points and high values
    land = form_land_mask_connect(elev, edry=2) >= 1
    high = form_land_mask_connect(elev, edry=8) >= 1
    
    hmat = np.full(
        (elev.shape[:]), hmax, dtype=spac.FLT32_t)
        
    hmat[land] = hmax
    if (hbeta > 0.0):
        hmat = np.minimum(
                hmat, tanh_wavelength_spacing(
                    elev, hmin, hmax, hcenter, hbeta))
    hmat[high] = hmax
    
    hmat = setup_shoreline_pixels(hmat, land, hmin)
    #-- and a little nonlinear smoothing
        
    filt = filter_pixels_harmonic(hmat, exp=2)
    hmat = np.minimum(hmat, filt)
        
    filt = filter_pixels_harmonic(hmat, exp=1)
    hmat = np.minimum(hmat, filt)
    
    hmat = np.asarray(remap_pixels_to_corner(hmat), 
                      dtype=spac.FLT32_t)
    
    #-- pack h(x) data to jigsaw datatype: average pixel-to-
    #-- node, careful with periodic BCs.
        
    spac.mshID = "ellipsoid-grid"
    spac.radii = geom.radii
    spac.xgrid = xlon * np.pi / 180.
    spac.ygrid = ylat * np.pi / 180.
    
    spac.value = hmat
    spac.slope = np.array(dhdx)
    spac.value = np.minimum(hmax, spac.value)
    
    jigsawpy.savemsh(opts.hfun_file, spac)

    #smooth hmat
    jigsawpy.cmd.marche(opts, spac)
    
    # make mesh using JIGSAW-----------------------------------------------
    opts.hfun_scal = "absolute"
    opts.hfun_hmax = float("inf")           # global maximum mesh resolution (similar to hmax)
    opts.hfun_hmin = float(+0.00)           # global maximum mesh resolution (similar to hmax)
    opts.mesh_dims = +2             # 2-dim. simplexes
    opts.optm_iter = +64           # number of itereation for the optimization
    opts.optm_kern = "cvt+dqdx"
    opts.optm_cost = "skew-cos"
    opts.optm_qlim = +9.5E-01
    opts.optm_qtol = +1.0E-05
    opts.optm_tria = True
    opts.optm_dual = False
#    # make mesh using JIGSAW-----------------------------------------------
#    opts.hfun_scal = "absolute"
#    opts.hfun_hmax = float("inf")       # null HFUN limits
#    opts.hfun_hmin = float(+0.00)
#    #opts.hfun_hmax = 1.25*Smax       # Unintended effects, better off null 
#    #opts.hfun_hmin = .5*Smin
#    opts.optm_iter = +64            # number of itereation for the optimization
#    opts.optm_cost = "skew-cos"
#    
#    opts.mesh_dims = +2                 # 2-dim. simplexes
#    
#    #opts.mesh_top1 = "true" !!!No convergece
    
    
    ttic = time.time()
    
    jigsawpy.cmd.jigsaw(opts, mesh)
    
    ttoc = time.time()
    
    print("CPUSEC =", (ttoc - ttic))
    
    jigsawpy.savemsh(meshFile, mesh)


from FilterRoutinesNM import *

# load mesh
jigsawpy.loadmsh(meshFile, mesh) # uncomment if starting here 
# Now apply filters and output in WW3 form

opts.geom_file = os.path.join(outDir, "geom.msh")  #jigsaw ctlr file
opts.jcfg_file = os.path.join(outDir, "opts.jig")  #jigsaw ctlr file
   
geom.mshID = "ellipsoid-mesh"
geom.radii = np.full(3, 6.371E+003, dtype=geom.REALS_t)

jigsawpy.savemsh(opts.geom_file, geom)

inject_dem()
filter_ocn()
 
# viz. in eg. paraview
#jigsawpy.savevtk(OutDir+"test.vtk", mesh)

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

write_gmsh_mesh(os.path.join(outDir,"GDWPS.ww3"), point, tri_data)
write_2dm_mesh(os.path.join(outDir,"GDWPS.2dm"), point, tri_data)

#mesh.point["coord"]=point
#jigsawpy.savemsh(os.path.join(outDir,"NWA.F.LLH.msh"), mesh)
#
#
##write final mesh in jigsaw .msh format
#meshR2 = jigsawpy.jigsaw_msh_t()
##make 2D coordinates 
#nd=point.shape
#meshR2.ndims=2
#meshR2.vert2 = np.zeros(nd[0], dtype=mesh.VERT2_t)
#meshR2.vert2["coord"] = point[:,[0,1]]
#meshR2.tria3=mesh.tria3
#meshR2.mshID=mesh.mshID
#
#jigsawpy.savemsh(os.path.join(outDir, "NWA.F.LL.msh"), meshR2)
    
