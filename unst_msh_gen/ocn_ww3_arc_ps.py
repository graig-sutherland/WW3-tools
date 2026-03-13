
""" Jigsaw meshes for WW3 with global bathymetry
"""

# Authors: Ali Salimi, Darren

# The DEM file used below can be found at:	
# https://github.com/dengwirda/dem/releases/tag/v0.1.1
import argparse
import numpy as np
import netCDF4 as nc
import os

import jigsawpy

from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from spacing import *

# Create the parser
#parser = argparse.ArgumentParser(description="Run mesh gen with specific region settings.")
#
## Add an argument for the black_sea option
#parser.add_argument('--black_sea', type=int, default=3,
#                    help='Set the region mode: 1 no black-sea, 2 black-sea detached, 3 black-sea with connections. Default is 3.')
#
## Parse the command-line arguments
#args = parser.parse_args()

# Use args.black_sea to access the black_sea value



ISOLATED = 30000.  # min surface area [km^2]

# just global objects, to keep things simple...
geom = jigsawpy.jigsaw_msh_t()
spac = jigsawpy.jigsaw_msh_t()
mesh = jigsawpy.jigsaw_msh_t()
opts = jigsawpy.jigsaw_jig_t()

def create_msh():

#-- create a simple uniform mesh for the globe

    print("*create-msh...")

    opts.geom_file = "geom.msh"
    opts.hfun_file = "spac.msh"
    opts.jcfg_file = "opts.jig"
    
    geom.mshID = "euclidean-mesh"
    # going to read in mesh dimensions
    gebcoDir = '/home/gsu000/l5/GEBCO'
    bathyFile = os.path.join(gebcoDir, 'IBCAO_v4_2_13_400m_ice.nc')
#    bathyFile = "RTopo_2_0_4_GEBCO_v2023_60sec_pixel.nc"
    data = nc.Dataset(bathyFile, "r")
    xgrid = np.asarray(data["x"][:])
    ygrid = np.asarray(data["y"][:])
    geom.vert2 = np.array([
        ((xgrid.min(),ygrid.min()), 0),
        ((xgrid.max(),ygrid.min()), 0),
        ((xgrid.max(),ygrid.max()), 0),
        ((xgrid.min(),ygrid.max()), 0)],
        dtype=geom.VERT2_t)

    geom.edge2 = np.array([   # list of "edges" between vert
        ((0, 1), 0),          # outer square
        ((1, 2), 0),
        ((2, 3), 0),
        ((3, 0), 0)],
        dtype=geom.EDGE2_t)

#    geom.radii = np.full(
#        3, 6.371E+003, dtype=geom.REALS_t)

    jigsawpy.savemsh(opts.geom_file, geom)
    
    create_siz()
    
    jigsawpy.savemsh(opts.hfun_file, spac)
    
    # solve |dh/dx| constraints in spacing
    jigsawpy.cmd.marche(opts, spac)
    
    opts.mesh_file = "mesh_ARC_5km_PS.msh"
    
    opts.hfun_scal = "absolute"
    opts.hfun_hmax = +5000.           # uniform at 5.km
#    opts.hfun_hmin  = +5.
    opts.mesh_dims = +2             # 2-dim. simplexes
    opts.optm_iter = +0
#    opts.optm_cost = "skew-cos"

    jigsawpy.cmd.jigsaw(opts, mesh)

    
def create_siz():

#-- create mesh spacing function for the Arctic

    hmax = 5000.0 # maximum spacing [km]
    hmin = hmax  # minimum spacing
    hshr = hmax    # shoreline spacing
    nwav = 0.  # number of cells per sqrt(g*H)
    dhdx = 0.0  # allowable spacing gradient

    gebcoDir = '/home/gsu000/l5/GEBCO'
    bathyFile = os.path.join(gebcoDir, 'IBCAO_v4_2_13_400m_ice.nc')
#    bathyFile = "RTopo_2_0_4_GEBCO_v2023_60sec_pixel.nc"
    data = nc.Dataset(bathyFile, "r")
    xgrid = np.asarray(data["x"][:])
    ygrid = np.asarray(data["y"][:])
    elev = np.asarray(data["z"][:])
           
    land = form_land_mask_connect(elev, edry=-10) >= 1
    high = form_land_mask_connect(elev, edry=0) >= 1

#-- init. h(x) data: impose global "reachable" land mask

    hmat = np.full(
        (elev.shape[:]), hmax, dtype=spac.FLT32_t)
#-- h(x) data equal to sqrt(elevation)
#    hmat = np.sqrt(np.maximum(-elev,0)).astype(spac.FLT32_t)
#-- elevation gradient spacing
#    nslp = 1
#    sdev = 2
#    hmat = elev_sharpness_spacing(xgrid, ygrid, elev, dhdx, land, nslp, hmin, hmax, sdev)
        
    hmat[land] = hmax

    if (nwav > 0.0):
        hmat = np.minimum(
            hmat, swe_wavelength_spacing(
                elev, land, nwav, hmin, hmax))
    
#-- final h(x) data: impose global "shoreline" min. val.
   
    hmat[high] = hmax
   
    hmat = setup_shoreline_pixels(hmat, land, hshr)

#-- and a little nonlinear smoothing
    
    filt = filter_pixels_harmonic(hmat, exp=2)
    hmat = np.minimum(hmat, filt)
        
    filt = filter_pixels_harmonic(hmat, exp=1)
    hmat = np.minimum(hmat, filt)

    hmat = np.asarray(remap_pixels_to_corner(hmat), 
                      dtype=spac.FLT32_t)
    
#-- pack h(x) data to jigsaw datatype: average pixel-to-
#-- node, careful with periodic BCs.
    
    spac.mshID = "euclidean-mesh"
    spac.xgrid = xgrid # keep in m
    spac.ygrid = ygrid

    xmat, ymat = np.meshgrid(
        spac.xgrid, spac.ygrid, sparse=True)

        
    spac.value = hmat
    spac.slope = np.array(dhdx)
    spac.value = np.maximum(hmin, spac.value)
    spac.value = np.minimum(hmax, spac.value)
    print(f'min/max spacing is {spac.value.min():.1f}/{spac.value.max():.1f}')
    
#-- save spacing to a netcdf, for viz. in e.g. paraview
    
#    data = nc.Dataset("spac.nc", "w")
#    data.createDimension("nlon", spac.xgrid.size)
#    data.createDimension("nlat", spac.ygrid.size) 
#
#    if ("val" not in data.variables.keys()):
#        data.createVariable("val", "f4", ("nlat", "nlon"))
#
#    data["val"][:, :] = spac.value[:, :]
#    data.close()
    

def inject_dem():

#-- remap a DEM on to the vertices of the mesh

    print("*inject-dem...")

#    data = nc.Dataset(
#        "RTopo_2_0_4_GEBCO_v2023_60sec_pixel.nc", "r")
    gebcoDir = '/home/gsu000/l5/GEBCO'
    bathyFile = os.path.join(gebcoDir, 'IBCAO_v4_2_13_400m_ice.nc')
    data = nc.Dataset(bathyFile, "r")
    xgrid = np.asarray(data["x"][:])
    ygrid = np.asarray(data["y"][:])
    elev = np.asarray(data["z"][:])

    xmid = xgrid
    ymid = ygrid
    # set positive elevation to 0
    elev[elev>0] = 0.
        
        
    ffun = RegularGridInterpolator(
        (ymid, xmid), elev, 
        bounds_error=False, fill_value=None)

    vert = mesh.point["coord"]
    mids =(vert[mesh.tria3["index"][:, 0], :] +
           vert[mesh.tria3["index"][:, 1], :] +
           vert[mesh.tria3["index"][:, 2], :] 
          ) / 3.0 

#    vsph = jigsawpy.R3toS2(geom.radii, vert)
#    vsph*= 180. / np.pi
    
    mesh.value = ffun((vert[:, 1], vert[:, 0]))
    
#    msph = jigsawpy.R3toS2(geom.radii, mids)
#    msph*= 180. / np.pi

    mesh.vmids = ffun((mids[:, 1], mids[:, 0]))
    
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
        mesh.point["coord"], 
        mesh.tria3["index"][mask, :])

    for iprt in range(nprt):
        itri = np.argwhere(part == iprt)
        asum = np.sum(area[itri])
       #print(asum)
        if asum < ISOLATED: mask[tris[itri]] = False
    
    # otherwise mark isolated cell as dry
    mask[tris[isol]] = False
    
    return mask


def filter_ocn():#black_sea=args.black_sea):

#-- use the remapped elev. to keep ocean cells

    print("*filter-ocn...")

    elev =(mesh.value[mesh.tria3["index"][:, 0]]
         + mesh.value[mesh.tria3["index"][:, 1]]
         + mesh.value[mesh.tria3["index"][:, 2]]
         + mesh.vmids) / 4.0
    
    """
# Print the elevations in the additional region
    additional_elev = elev[np.logical_and.reduce((
        mesh.smids[:, 1] >= additional_lat_min,
        mesh.smids[:, 1] <= additional_lat_max,
        mesh.smids[:, 0] >= additional_lon_min,
        mesh.smids[:, 0] <= additional_lon_max
    ))]
    print("Elevations in the additional region:")
    print(additional_elev)

# Print the elevations in the additional region2
    additional_elev2 = elev[np.logical_and.reduce((
        mesh.smids[:, 1] >= additional_lat_min2,
        mesh.smids[:, 1] <= additional_lat_max2,
        mesh.smids[:, 0] >= additional_lon_min2,
        mesh.smids[:, 0] <= additional_lon_max2
    ))]
    print("Elevations in the additional region2:")
    print(additional_elev2)

# Print the elevations in the additional region3
    additional_elev3 = elev[np.logical_and.reduce((
        mesh.smids[:, 1] >= additional_lat_min3,
        mesh.smids[:, 1] <= additional_lat_max3,
        mesh.smids[:, 0] >= additional_lon_min3,
        mesh.smids[:, 0] <= additional_lon_max3
    ))]
    print("Elevations in the additional region3:")
    print(additional_elev3)

    """

    # zssh, to cull elev. against
    surf = -10*np.ones(elev.shape, dtype=np.float32)
    # Update the surf array to include both regions
        
    keep = elev <= surf  # only keep tri with wet elev
   
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
    
    irev = np.zeros(mesh.point.size, dtype=np.int32)
    irev[ifwd] = np.arange(ifwd.size, dtype=np.int32)

    mesh.point = mesh.point[ifwd]
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

    create_msh()
    inject_dem()
    filter_ocn()
    
    # viz. in eg. paraview
    jigsawpy.savevtk("ARC_5km_PS.vtk", mesh)
    
    point = mesh.point["coord"]
#    point = jigsawpy.R3toS2(geom.radii, point)  # to [lon,lat] in deg
#    point*= 180. / np.pi
    depth = np.reshape(-1*mesh.value, (mesh.value.size, 1))
    #depth[depth < 0] = 50
    point = np.hstack((point, depth))  # append elev. as 3rd coord.
    cells = [("triangle", mesh.tria3["index"])]
    tri_data=cells[0][1]+1
    write_gmsh_mesh("ARC_5km_PS.ww3", point, tri_data)
   
