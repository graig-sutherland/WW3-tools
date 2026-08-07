""" Jigsaw meshes for WW3 with global bathymetry
"""

# Authors: Ali Salimi-Tarazouj, Darren Engwirda

# The DEM file used below can be found at:	
# https://github.com/dengwirda/dem/releases/tag/v0.1.1
import os
import argparse
import configparser
import numpy as np
import netCDF4 as nc

import jigsawpy

import geopandas as gpd
import pandas as pd
from geopandas.tools import sjoin
from shapely.geometry import MultiPoint, LineString, Polygon, MultiPolygon, mapping
import rioxarray
import xarray as xr

from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from spacing import *

def parse_input_args():
    parser = argparse.ArgumentParser(description='Create a mask file with multiple methods.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    return args

def load_configuration(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)

        # Creating a dictionary and populating it with configuration settings
    configurations = {
        'mesh_file': config.get('MeshSettings', 'mesh_file', fallback=''),
        'ww3_mesh_file':config.get('MeshSettings', 'WW3_mesh_file', fallback=''),
        'hfun_hmax': float(config.get('MeshSettings', 'hfun_hmax', fallback='100')),
        'hfun_hmin': float(config.get('MeshSettings', 'hfun_hmin', fallback='100')),
        'mask_file': config.get('CommandLineArgs', 'mask_file', fallback=''),
        'hmax': float(config.get('Spacing', 'hmax', fallback='100.0')),
        'hshr': float(config.get('Spacing', 'hshr', fallback='100')),
        'nwav': int(config.get('Spacing', 'nwav', fallback='400')),
        'hmin': float(config.get('Spacing', 'hmin', fallback='100.0')),
        'dhdx': float(config.get('Spacing', 'dhdx', fallback='0.05')),
        'dem_file': config.get('DataFiles', 'dem_file', fallback=''),
        'geom_file': config.get('DataFiles', 'geom_file', fallback=''),
        'PolarStereographic':bool(config.get('DataFiles','PolarStereographic', fallback=0))
    }
    return configurations

def great_circle(R, lon1, lat1, lon2, lat2):
    '''
    returns great circle distance using spherical law of cosines
    '''
    dlon = lon2 - lon1
    distance = R * np.arccos(np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(dlon))
    return distance

ISOLATED = 30000.*1e0  # min surface area [km^2]

# just global objects, to keep things simple...
geom = jigsawpy.jigsaw_msh_t()
geom3 = jigsawpy.jigsaw_msh_t()
spac = jigsawpy.jigsaw_msh_t()
mesh = jigsawpy.jigsaw_msh_t()
opts = jigsawpy.jigsaw_jig_t()
proj = jigsawpy.jigsaw_prj_t()

def create_msh():

#-- create a simple uniform mesh for the globe

    
    args = parse_input_args()
    configurations = load_configuration(args.config)

    geom_file = configurations["geom_file"] #"/home/gsu000/data/ppp7/RDWPS/NWA_geom_outside.msh"
    PolarStereographic = configurations['PolarStereographic']

    print("*create-msh...")
    if PolarStereographic:
        print(f' ... using polar stereographic grid')

    opts.geom_file = "./data/geom.msh"  #saves the geometry info for jigsaw
    opts.geom3_file = "./data/geom3.msh"  #saves the geometry info for jigsaw
    opts.hfun_file = "./data/spac.msh"  #saves the final mesh spacing info 
    opts.jcfg_file = "./data/opts.jig"  #jigsaw ctlr file
    opts.topo_file = "./data/topo.msh"
    opts.mask_file = "./data/mask.msh"
    opts.mesh_file = configurations['mesh_file']  #jigsaw format mesh file
    
    geom3.mshID = "ellipsoid-mesh"
    geom3.radii = np.full(
        3, 6.371E+003, dtype=geom.REALS_t)
    jigsawpy.savemsh(opts.geom3_file, geom3)
    jigsawpy.loadmsh(geom_file, geom)
    jigsawpy.savemsh(opts.geom_file, geom)
    
    create_siz()

    #------------------------------------ do stereographic proj.
    # truncate data to bounding rectangle of input PSLG--------------------
    xmin = np.min( geom.point["coord"][:, 0])
    ymin = np.min( geom.point["coord"][:, 1])
    xmax = np.max( geom.point["coord"][:, 0])

    proj.prjID = 'stereographic'
    proj.radii = +6.371E+003
    proj.xbase = +0.500 * (xmin + xmax) * np.pi / 180.
    if PolarStereographic:
        ymax=90.0
        proj.ybase = np.pi / 2.0
    else:
        ymax = np.max( geom.point["coord"][:, 1])
        proj.ybase = +0.500 * (ymin + ymax) * np.pi / 180.

    geom.point["coord"][:, :] *= np.pi / 180.
    jigsawpy.project(geom, proj, "fwd")
    jigsawpy.project(spac, proj, "fwd")

    jigsawpy.savemsh(opts.geom_file, geom)
    jigsawpy.savemsh(opts.hfun_file, spac)
    
    # solve |dh/dx| constraints in spacing
    jigsawpy.cmd.marche(opts, spac)
    
    opts.hfun_scal = "absolute"
    opts.hfun_hmax = configurations['hfun_hmax']           # global maximum mesh resolution (similar to hmax)
    opts.hfun_hmin = configurations['hfun_hmin']           # global maximum mesh resolution (similar to hmax)
    opts.mesh_dims = +2             # 2-dim. simplexes
    opts.optm_iter = +64           # number of itereation for the optimization
    opts.optm_kern = "cvt+dqdx"
    opts.optm_cost = "skew-cos"
    opts.optm_qlim = +9.5E-01
    opts.optm_qtol = +1.0E-05
    opts.optm_tria = True
    opts.optm_dual = False
    opts.verbosity = +1

    jigsawpy.cmd.jigsaw(opts, mesh)

#    # inverse projection
    jigsawpy.project(mesh, proj, "inv")
    jigsawpy.project(geom, proj, "inv")
    jigsawpy.project(spac, proj, "inv")
    
    # transform mesh nodes to degree lat, lon
    mesh.point["coord"][:, :] = mesh.point["coord"][:, :]*180. / np.pi
    
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
    meshR3.radii = geom3.radii
    # save
    jigsawpy.savemsh(opts.mesh_file, meshR3)
    
def create_siz():

    args = parse_input_args()
    configurations = load_configuration(args.config)

    #-- create mesh spacing function for the globe: for uniform mesh hmax = hshr = hmin

    hmax = configurations['hmax'] # maximum spacing [km] 
    hshr = configurations['hshr']   # shoreline spacing
    nwav = configurations['nwav']   # number of cells per sqrt(g*H)
    hmin = configurations['hfun_hmin']  # minimum spacing
    dhdx = configurations['dhdx']  # allowable spacing gradient: for more gradual transition use lower value
    mask_file = configurations['mask_file'] #user defined scaling file
    # Load the DEM file from the config
    dem_file = configurations['dem_file']
    print(f' ... dem file is {dem_file}')
    PolarStereographic = configurations['PolarStereographic']

    data = nc.Dataset(dem_file,"r")

    xlon = np.asarray(data["lon"][:])
    ylat = np.asarray(data["lat"][:])
    elev = np.asarray(data["bed_elevation"][:]) + \
           np.asarray(data["ice_thickness"][:])
           
    land = form_land_mask_connect(elev, edry=2) >= 1
    high = form_land_mask_connect(elev, edry=8) >= 1

#-- init. h(x) data: impose global "reachable" land mask

    hmat = np.full(
        (elev.shape[:]), hmax, dtype=spac.FLT32_t)
        
    hmat[land] = hmax

    if (nwav > 0.0):
        print(f"Calculating depth dependent size for nwave = {nwav}")
        hmat = np.minimum(
            hmat, swe_wavelength_spacing(
                elev, land, nwav, hmin, hmax))
    
#-- final h(x) data: impose global "shoreline" min. val.
   
    hmat[high] = hmax
   
    hmat = setup_shoreline_pixels(hmat, land, hshr)
   
#-- apply user-defined scaling: multiply h(x) by mask array
    if mask_file:
        print(f"create_siz: Scaling applied using mask_file: {mask_file}")
        ds = nc.Dataset(mask_file, "r")
        hmat = np.asarray(ds["val"][:,:], dtype=float)
        ds.close()
        #hmat = scale_spacing_via_mask(args, hmat)
    else:
    # Handle case where mask_file is not provided
        print("create_siz: No mask file provided. Proceeding without scaling...")


#-- a little nonlinear smoothing
    
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
    if PolarStereographic:
        ymax = 90.0
    else:
        ymax = np.max( geom.point["coord"][:, 1])

    # create spac msh 
    xmsk = np.logical_and( xlon > xmin , xlon < xmax )
    ymsk = np.logical_and( ylat > ymin , ylat < ymax )

    xlons, ylats = xlon[xmsk], ylat[ymsk]
    hmats = hmat[ymsk,:]
    hmats = hmats[:,xmsk]
    
#-- pack h(x) data to jigsaw datatype: average pixel-to-
#-- node, careful with periodic BCs.
    
    spac.mshID = "ellipsoid-grid"
    spac.radii = geom3.radii
    spac.xgrid = xlons * np.pi / 180.
    spac.ygrid = ylats * np.pi / 180.
    spac.value = hmats
    
    spac.slope = np.array(dhdx)
    spac.value = np.maximum(hmin, spac.value)

    
#-- save spacing to a netcdf, for viz. in e.g. paraview
    
#`    data = nc.Dataset("./data/spac.nc", "w")
#`    data.createDimension("nlon", spac.xgrid.size)
#`    data.createDimension("nlat", spac.ygrid.size) 
#`
#`    if ("val" not in data.variables.keys()):
#`        data.createVariable("val", "f4", ("nlat", "nlon"))
#`    if ("lon" not in data.variables.keys()):
#`        data.createVariable("lon", "f4", ("nlon"))
#`    if ("lat" not in data.variables.keys()):
#`        data.createVariable("lat", "f4", ("nlat"))
#`
#`    data["lon"][:] = spac.xgrid*180/np.pi
#`    data["lat"][:] = spac.ygrid*180/np.pi
#`    data["val"][:, :] = spac.value[:, :]
#`    data.close()
    
def inject_mask():

    args = parse_input_args()
    configurations = load_configuration(args.config)

#-- remap a DEM on to the vertices of the mesh

    print("*inject-gshhs-mask...")

        # Load the DEM file from the config
#    gsshs_file = configurations['dem_file']
    mask_spacing = "1m"          # 0.1 degree resolution
    res = 'f'                 # Intermediate resolution
    dem_file = f"../data/gshhs_mask_{res}_{mask_spacing}_ocean.nc"
    ds = xr.open_dataset(dem_file)
    da = ds["landmask"]
    xmid = da.coords["lon"].to_numpy()
    ymid = da.coords["lat"].to_numpy()
    landmask = da.to_numpy()
    # define spatial dimensions and crs in data array
#    da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
#    da.rio.write_crs("epsg:4326", inplace=True)

    # set values outside region to 1
#    geom_file = configurations["geom_file"] #"/home/gsu000/data/ppp7/RDWPS/NWA_geom_outside.msh"
#    # define base geoseries. I have coordinates from Patrick
#    rasterDir = '/home/gsu000/data/ppp7/GEBCO'
#    glon = np.genfromtxt(os.path.join(rasterDir, 'nwa5km.lon'))
#    glat = np.genfromtxt(os.path.join(rasterDir, 'nwa5km.lat'))
#    bbox_lon = np.concatenate((glon[0,::-1], glon[:,0].T, glon[-1,:], glon[::-1,-1]))
#    bbox_lon = np.mod(bbox_lon+180, 360) - 180.
#    bbox_lat = np.concatenate((glat[0,::-1], glat[:,0].T, glat[-1,:], glat[::-1,-1]))
#    boundary_coords = np.vstack((bbox_lon,bbox_lat)).T
#    # convert to geopandas series
#    search_area = Polygon(boundary_coords)
#    bound_gdf = gpd.GeoDataFrame(index=[0], crs="epsg:4326", geometry=[search_area])
#
#    # clip
#    clipped = da.rio.clip(bound_gdf.geometry.apply(mapping), bound_gdf.crs, drop=False)
#    # fillna with 0
#    landmask = clipped.fillna(1).to_numpy()

        
    ffun = RegularGridInterpolator(
        (ymid, xmid), landmask, 
        bounds_error=False, fill_value=None)


    vert = mesh.vert3["coord"]
    mids =(vert[mesh.tria3["index"][:, 0], :] +
           vert[mesh.tria3["index"][:, 1], :] +
           vert[mesh.tria3["index"][:, 2], :] 
          ) / 3.0 

    vsph = jigsawpy.R3toS2(geom3.radii, vert)
    vsph*= 180. / np.pi
    
    mesh.value = ffun((vsph[:, 1], vsph[:, 0]))
    
    msph = jigsawpy.R3toS2(geom3.radii, mids)
    msph*= 180. / np.pi

#    mesh.vmids = ffun((msph[:, 1], msph[:, 0]))
    
    # save lon-lat at cell centres
    mesh.smids = np.zeros(
        (mesh.tria3.size, 2), dtype=np.float64)
    mesh.smids[:, 0] = msph[:, 0]
    mesh.smids[:, 1] = msph[:, 1]

def inject_dem():

    args = parse_input_args()
    configurations = load_configuration(args.config)

#-- remap a DEM on to the vertices of the mesh

    print("*inject-dem...")

        # Load the DEM file from the config
    dem_file = configurations['dem_file']

    data = nc.Dataset(dem_file,"r")

    xlon = np.asarray(data["lon"][:])
    ylat = np.asarray(data["lat"][:])
    elev = np.asarray(data["bed_elevation"][:]) + \
           np.asarray(data["ice_thickness"][:])
        
    xmid = 0.5 * (xlon[:-1:] + xlon[1::])
    ymid = 0.5 * (ylat[:-1:] + ylat[1::])
        
    ffun = RegularGridInterpolator(
        (ymid, xmid), elev, 
        bounds_error=False, fill_value=None)

    vert = mesh.vert3["coord"]
    mids =(vert[mesh.tria3["index"][:, 0], :] +
           vert[mesh.tria3["index"][:, 1], :] +
           vert[mesh.tria3["index"][:, 2], :] 
          ) / 3.0 

    vsph = jigsawpy.R3toS2(geom3.radii, vert)
    vsph*= 180. / np.pi
    
    mesh.value = ffun((vsph[:, 1], vsph[:, 0]))
    
    msph = jigsawpy.R3toS2(geom3.radii, mids)
    msph*= 180. / np.pi

#    mesh.vmids = ffun((msph[:, 1], msph[:, 0]))
    
    # save lon-lat at cell centres
    mesh.smids = np.zeros(
        (mesh.tria3.size, 2), dtype=np.float64)
    mesh.smids[:, 0] = msph[:, 0]
    mesh.smids[:, 1] = msph[:, 1]


def tri_to_tri(tria):

#-- return tria-to-tria adj. as a sparse graph

    # non-unique edges in tris
    edge = np.empty((0, 2), dtype=np.int32)
    tris = np.empty((0), dtype=np.int32)
    edge = np.concatenate((edge, 
        tria[:, (0, 1)]), axis=0)
    tris = np.concatenate((tris, 
        np.arange(0, tria.shape[0])))
        
    edge = np.concatenate((edge, 
        tria[:, (1, 2)]), axis=0)
    tris = np.concatenate((tris, 
        np.arange(0, tria.shape[0])))
        
    edge = np.concatenate((edge, 
        tria[:, (2, 0)]), axis=0)
    tris = np.concatenate((tris, 
        np.arange(0, tria.shape[0])))
        
    # which edges match to which?
    edge = np.sort(edge, axis=1)
    imap = np.argsort(edge[:, 1], kind="stable")
    edge = edge[imap, :]
    tris = tris[imap]
    imap = np.argsort(edge[:, 0], kind="stable")
    edge = edge[imap, :]
    tris = tris[imap]
    
    diff = edge[1::, :] - edge[:-1:, :]

    same = np.argwhere(np.logical_and.reduce((
        diff[:, 0] == 0, 
        diff[:, 1] == 0))).ravel()
        
    # tris[same] and tris[same+1] share
    rows = np.concatenate((
        tris[same], tris[same+1]))
    cols = np.concatenate((
        tris[same+1], tris[same]))
    data = np.ones(rows.size, dtype=np.int8)
    
    # ith tri is adj. to tri in ith row
    return csr_matrix((data, (rows, cols)))


def filter_dry(mesh, mask):

#-- require dry cells > 1 dry edge, and large

    print("*filter-dry...")

    filt = np.logical_not(mask)
    tris = np.argwhere(filt).ravel()

    conn = tri_to_tri(mesh.tria3["index"][filt, :])
        
    # require dry to be adj. >=1 dry cell
    isol = np.sum(conn, axis=1) <= 1
    isol = np.ravel(isol)

    """
    # delete groups of dry if too small
    nprt, part = connected_components(
        conn, directed=False, return_labels=True)

    tris = np.argwhere(filt).ravel()
    for iprt in range(nprt):
        itri = np.argwhere(part == iprt)
        if (itri.size <= 2): mask[tris[itri]] = True
    """

    # otherwise mark isolated cell as ocn
    mask[tris[isol]] = True

    return mask


def filter_wet(mesh, mask):

#-- require wet cells > 1 wet edge, and large

    print("*filter-wet...")

    tris = np.argwhere(mask).ravel()

    conn = tri_to_tri(mesh.tria3["index"][mask, :])

    # require wet to be adj. >=1 wet cell
    isol = np.sum(conn, axis=1) <= 1
    isol = np.ravel(isol)
    
    # delete groups of wet if too small
    nprt, part = connected_components(
        conn, directed=False, return_labels=True)

    area = jigsawpy.trivol2(
        mesh.vert3["coord"], 
        mesh.tria3["index"][mask, :])

    for iprt in range(nprt):
        itri = np.argwhere(part == iprt)
        asum = np.sum(area[itri])
       #print(asum)
        if asum < ISOLATED: mask[tris[itri]] = False
    
    # otherwise mark isolated cell as dry
    mask[tris[isol]] = False
    
    return mask


def filter_ocn():

    args = parse_input_args()
    configurations = load_configuration(args.config)

#-- use the gshhs data to filter dry cells NB: currently slow. will try with a raster map

    print("*filter-ocn...")

    mask =(mesh.value[mesh.tria3["index"][:, 0]]
         + mesh.value[mesh.tria3["index"][:, 1]]
         + mesh.value[mesh.tria3["index"][:, 2]]) / 3.
    keep = mask < 0.15 # water cells
    
#    elev =(mesh.value[mesh.tria3["index"][:, 0]]
#         + mesh.value[mesh.tria3["index"][:, 1]]
#         + mesh.value[mesh.tria3["index"][:, 2]]) / 3.0
#    surf = np.zeros(elev.shape, dtype=np.float32)
#    keep = elev <= surf  # only keep tri with wet elev
    # iterate on dry cells until none "isolated" 
    knum = np.count_nonzero(keep)
    while (True):   
        keep = filter_dry(mesh, keep)
        if (np.count_nonzero(keep) == knum): break
        knum = np.count_nonzero(keep)
    
    mesh.tria3 = mesh.tria3[keep]

    # iterate on wet cells until none "isolated"
    keep = np.ones(mesh.tria3.size, dtype=bool)
    knum = np.count_nonzero(keep)
    while (True):
        keep = filter_wet(mesh, keep)
        if (np.count_nonzero(keep) == knum): break
        knum = np.count_nonzero(keep)
    
    mesh.tria3 = mesh.tria3[keep]

    # delete unused vertices and reindex
    ifwd = np.unique(mesh.tria3["index"].ravel())
    
    irev = np.zeros(mesh.vert3.size, dtype=np.int32)
    irev[ifwd] = np.arange(ifwd.size, dtype=np.int32)

    mesh.vert3 = mesh.vert3[ifwd]
    mesh.value = mesh.value[ifwd]
    mesh.tria3["index"] = irev[mesh.tria3["index"]]
    return mesh

def filter_ocn_wetonly():
    # iterate on wet cells until none "isolated"
    keep = np.ones(mesh.tria3.size, dtype=bool)
    knum = np.count_nonzero(keep)
    while (True):
        keep = filter_wet(mesh, keep)
        if (np.count_nonzero(keep) == knum): break
        knum = np.count_nonzero(keep)
    
    mesh.tria3 = mesh.tria3[keep]

    # delete unused vertices and reindex
    ifwd = np.unique(mesh.tria3["index"].ravel())
    
    irev = np.zeros(mesh.vert3.size, dtype=np.int32)
    irev[ifwd] = np.arange(ifwd.size, dtype=np.int32)

    mesh.vert3 = mesh.vert3[ifwd]
    mesh.value = mesh.value[ifwd]
    mesh.tria3["index"] = irev[mesh.tria3["index"]]
    return mesh

def write_gmsh_mesh(filename, node_data, tri):
    num_nodes = node_data.shape[0]

    # Write the mesh to the file
    with open(filename, 'w') as fileID:
        fileID.write("$MeshFormat\n")
        fileID.write("2 0 8\n")
        fileID.write("$EndMeshFormat\n")
        fileID.write("$Nodes\n")
        fileID.write(str(num_nodes) + "\n")

        for i in range(num_nodes):
            fileID.write(
                f"{i + 1}  {node_data[i, 0]:5.5f} {node_data[i, 1]:5.5f} {node_data[i, 2]:5.5f}\n"
            )

        fileID.write("$EndNodes\n")
        fileID.write("$Elements\n")
        num_elements = len(tri)
        fileID.write(str(num_elements) + "\n")

        m = 0
        for i in range(len(tri)):
            m += 1
            fileID.write(f"{m} 2 3 0 {i+1} 0 {tri[i][0]} {tri[i][1]} {tri[i][2]}\n")

        fileID.write("$EndElements\n")



if (__name__ == "__main__"):

    args = parse_input_args()
    configurations = load_configuration(args.config)

    create_msh()

    jigsawpy.loadmsh(opts.mesh_file, mesh)

    inject_mask()
    filter_ocn()
    inject_dem()
    filter_ocn_wetonly()
    
    # viz. in eg. paraview
    #jigsawpy.savevtk("./data/test.vtk", mesh)
    
    point = mesh.vert3["coord"]
    point = jigsawpy.R3toS2(mesh.radii, point)  # to [lon,lat] in deg
    point*= 180. / np.pi
    depth = np.reshape(-1*mesh.value, (mesh.value.size, 1))
    depth[depth <= 0] = 1
    point = np.hstack((point, depth))  # append elev. as 3rd coord.
    cells = [("triangle", mesh.tria3["index"])]
    tri_data=cells[0][1]+1
    ww3_mesh_file = configurations['ww3_mesh_file']
    write_gmsh_mesh(ww3_mesh_file, point, tri_data)
