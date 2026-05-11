
import os
import argparse
import time
import numpy as np
import netCDF4 as nc

import jigsawpy
from spacing import *

from scipy.interpolate import RegularGridInterpolator

#-------------------Input Files----------------------------------------
if __name__ == "__main__":
    dataDir = "/home/gsu000/data/ppp7/RDWPS"
    #GSSH coastline 
    #jigsaw .msh format Planer Straight Line Graph defining mesh outer boundary and coastline
    #PSLGFile="NWcoastal.PSLG.msh"
    #PSLGFile=os.path.join(dataDir, "test.msh")
    PSLGFile = "../NWA_geom.msh"
    #jigsaw gridded .msh format Distance to taget poings
    #DistanceToCoastFile="DFun.NWcoastal.PSLG.msh"
    #jigsaw gridded .msh format topography on same grid as distance
    #TopographyFile="Topo.DFun.NWcoastal.PSLG.msh"
    TopographyFile = "/home/gsu000/projects/WW3-tools/unst_msh_gen/RTopo_2_0_4_GEBCO_v2023_60sec_pixel.nc"
    
    # directory to write output files to
    # tanh spacing
    hbeta = 0.01 # tanh width # make negative to have uniform spacing
    hcenter = 50
    hmin=1.0
    hmax=5.0 # test uniform grid
    dhdx=0.2
    
    exp = "NWA_geom_5km_to_1km_h050_hb01"
    outDir=os.path.join(dataDir, f'{exp}_jigsawpy')
    # Create the output directory------------------------------------------
    if not os.path.exists(outDir): os.makedirs(outDir)
    meshFile = os.path.join(outDir, "NWA.R3.msh")
    mesh = jigsawpy.jigsaw_msh_t()

    # flag to check if I need to calculate the mesh or just apply filters
    calculate_mesh = False
        
    if calculate_mesh:
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
        proj = jigsawpy.jigsaw_prj_t()
        
        opts.geom_file = os.path.join(outDir,"geom.msh")
        opts.jcfg_file = os.path.join(outDir,"jcfg.jig")
        opts.mesh_file = os.path.join(outDir,"mesh.msh")
        opts.hfun_file = os.path.join(outDir,"spac.msh")
        # load input data------------------------------------------------------
        jigsawpy.loadmsh(PSLGFile, geom)
        #jigsawpy.loadmsh(DistanceToCoastFile, dist)
        #if TopographyFile.endswith(".msh"):
        #    jigsawpy.loadmsh(TopographyFile, topo)
        #elif TopographyFile.endswith(".nc"):
        
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
        
        # truncate data to bounding rectangle of input PSLG--------------------
        xmin = np.min( geom.point["coord"][:, 0])
        ymin = np.min( geom.point["coord"][:, 1])
        xmax = np.max( geom.point["coord"][:, 0])
        ymax = np.max( geom.point["coord"][:, 1])
        
        # create spac msh 
        xmsk = np.logical_and( xlon > xmin , xlon < xmax )
        ymsk = np.logical_and( ylat > ymin , ylat < ymax )
        
        xlons, ylats = xlon[xmsk], ylat[ymsk]
        #-- pack h(x) data to jigsaw datatype: average pixel-to-
        #-- node, careful with periodic BCs.
            
        spac.mshID = "ellipsoid-grid"
        spac.radii = geom.radii
        spac.xgrid = xlons * np.pi / 180.
        spac.ygrid = ylats * np.pi / 180.
        
        hmats = hmat[ymsk,:]
        hmats = hmats[:,xmsk]
        spac.value = hmats
        spac.slope = np.array(dhdx)
        spac.value = np.minimum(hmax, spac.value)
        
        
        #------------------------------------ do stereographic proj.
        geom.point["coord"][:, :] *= np.pi / 180.
        
        proj.prjID = 'stereographic'
        proj.radii = +6.371E+003
        proj.xbase = +0.500 * (xmin + xmax) * np.pi / 180.
        proj.ybase = +0.500 * (ymin + ymax) * np.pi / 180.
        
        jigsawpy.savemsh(os.path.join(outDir,'spac_noproj.msh'), spac)
        
        jigsawpy.project(geom, proj, "fwd")
        jigsawpy.project(spac, proj, "fwd")
        
        jigsawpy.savemsh(opts.geom_file, geom)
        jigsawpy.savemsh(opts.hfun_file, spac)
        
        # save hmat------------------------------------------------------------
        jigsawpy.savemsh(os.path.join(outDir,"spac_proj.msh"), spac)
        
        #smooth hmat
        jigsawpy.cmd.marche(opts, spac)
        
        # save smoothed hmat---------------------------------------------------
        jigsawpy.savemsh(os.path.join(outDir,"spac_proj1.msh"), spac)
        
        # make mesh using JIGSAW-----------------------------------------------
        opts.hfun_scal = "absolute"
        opts.hfun_hmax = float("inf")       # null HFUN limits
        opts.hfun_hmin = float(+0.00)
        #opts.hfun_hmax = 1.25*Smax       # Unintended effects, better off null 
        #opts.hfun_hmin = .5*Smin
        opts.optm_iter = +64            # number of itereation for the optimization
        opts.optm_cost = "skew-cos"
        
        opts.mesh_dims = +2                 # 2-dim. simplexes
        opts.mesh_eps1 = +1.
        
        #opts.mesh_top1 = "true" !!!No convergece
        
        
        ttic = time.time()
        
        jigsawpy.cmd.jigsaw(opts, mesh)
        
        ttoc = time.time()
        
        print("CPUSEC =", (ttoc - ttic))
        
        # save mesh in JIGSAW native projection based on input PSLG------------
        jigsawpy.savemsh(os.path.join(outDir,f"NWA.PROJ.msh"),mesh)
        
        # compute costa functions and save
        cost = jigsawpy.triscr2(mesh.point["coord"],mesh.tria3["index"])
        np.savetxt(os.path.join(outDir,"TriScr2.txt"),cost,"%f")
        print("TRISCR =", np.min(cost), np.mean(cost))
        
        cost = jigsawpy.pwrscr2(mesh.point["coord"], mesh.power, mesh.tria3["index"])
        np.savetxt(os.path.join(outDir, "PwrScr2.txt"),cost,"%f")
        print("PWRSCR =", np.min(cost), np.mean(cost))
        
        tbad = jigsawpy.centre2(mesh.point["coord"],mesh.power,mesh.tria3["index"])
        print("OBTUSE =",+np.count_nonzero(np.logical_not(tbad)))
        
        
        # project mesh nodes to radian lat, lon
        jigsawpy.project(mesh, proj, "inv") # This used to work
        #jigsawpy.savemsh("RWPS.radian.msh",mesh)
        
        # transform mesh nodes to degree lat, lon
        mesh.point["coord"][:, :] = mesh.point["coord"][:, :]*180. / np.pi
        
        # save mesh in degree lat, lon system
        jigsawpy.savemsh(os.path.join(outDir,"NWA.LL.msh"),mesh)
        
        # create jigsaw R3 mesh on global surface and save to 
        S2=mesh.point["coord"][:,[0,1]]
        S2=S2*np.pi/180.
        
        R3=jigsawpy.S2toR3(mesh.radii,S2)
        #np.savetxt("R3.txt",R3," %f ") # save 3D nodes
        
        meshR3 = jigsawpy.jigsaw_msh_t()
        mesh.mshID = 'ellipsoid-mesh'
        meshR3.tria3=mesh.tria3
        meshR3.ndims=3
        #make 3D coordinates 
        nd=R3.shape
        meshR3.vert3 = np.zeros(nd[0], dtype=mesh.VERT3_t)
        meshR3.vert3["coord"] = R3
        jigsawpy.savemsh(meshFile, meshR3)
  
    # load mesh
    jigsawpy.loadmsh(meshFile, mesh) # uncomment if starting here 
    # Now apply filters and output in WW3 form
    # import FilterRoutines.py
    
    from FilterRoutinesNM import *
    
    #replace mesh with R3 mesh, mesh->mesh R3
    
#    opts.geom_file = PSLGFile  #saves the geometry info for jigsaw
#    jigsawpy.loadmsh(PSLGFile, geom)
    opts.geom_file = os.path.join(outDir, "geom.msh")  #jigsaw ctlr file
    opts.jcfg_file = os.path.join(outDir, "opts.jig")  #jigsaw ctlr file
       
    geom.mshID = "ellipsoid-mesh"
    geom.radii = np.full(3, 6.371E+003, dtype=geom.REALS_t)
    
    jigsawpy.savemsh(opts.geom_file, geom)
    
    mesh = inject_dem(mesh)
    mesh = filter_ocn(mesh)
        
    jigsawpy.savemsh(os.path.join(outDir, "NWA.F.R3.msh"), mesh)
    
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
    
    write_gmsh_mesh(os.path.join(outDir,"NWA.ww3"), point, tri_data)
    
    mesh.point["coord"]=point
    jigsawpy.savemsh(os.path.join(outDir,"NWA.F.LLH.msh"), mesh)
    
    
    #write final mesh in jigsaw .msh format
    meshR2 = jigsawpy.jigsaw_msh_t()
    #make 2D coordinates 
    nd=point.shape
    meshR2.ndims=2
    meshR2.vert2 = np.zeros(nd[0], dtype=mesh.VERT2_t)
    meshR2.vert2["coord"] = point[:,[0,1]]
    meshR2.tria3=mesh.tria3
    meshR2.mshID=mesh.mshID
    
    jigsawpy.savemsh(os.path.join(outDir, "NWA.F.LL.msh"), meshR2)
        
