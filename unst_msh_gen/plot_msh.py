#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:17:57 2024

@author: Ali Salimi-Tarazouj

This script originally was developed by Steven Brus (@sbrus89) and revised by Ali Salimi-Tarazouj to work efficiently for very high resolution meshes
"""

import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.tri as tri
import matplotlib.ticker as mticker
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import argparse
import sys, os
from projector import stereo3

def ll2xy(lon, lat, R=6378.0, lon0=0.0, lat0=np.pi/2):
    # input in degrees
    dlon = lon - lon0
    k = 2*R / (1.0 + np.sin(lat0)*np.sin(lat) + np.cos(lat0)*np.cos(lat)*np.cos(dlon))
    # output
    x = k*np.cos(lat)*np.sin(dlon)
    y = k*np.cos(lat0)*np.sin(lat) - k*np.sin(lat0)*np.cos(lat)*np.cos(dlon)
    return x, y

def xy2ll(x, y, R=6378.0, lon0=0.0, lat0=90.0):
    lon0 *= np.pi/180.0
    lat0 *= np.pi/180.0
    p = np.sqrt(x*x + y*y)
    c = 2.0 * np.arctan2(2.0*R, p)
    #output
    lat = np.arcsin(np.cos(c)*np.sin(lat0) + y*np.sin(c)*np.cos(lat0)/p)
    dlon = np.arctan2(p*np.cos(lat0)*np.cos(c) - y*np.sin(lat0)*np.sin(c), x*np.sin(c))
    lon = lon0 + dlon
    return lon, lat

def add_map_features(ax):
    ax.coastlines()
    gl = ax.gridlines()
    ax.add_feature(cfeature.BORDERS);
#    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='k', facecolor=cfeature.COLORS['land']))
    gl = ax.gridlines()#draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)

def read_gmsh(filename):
    #purpose: this function reads a gmsh file and returns node and element information
    # additional information about the gmsh file format can be found
    # in many places including here: http://gmsh.info/dev/doc/texinfo/gmsh.pdf
    # The gmsh format is what the WW3 model uses to define unstructured grids

    #input:
    # filename - name of gmsh file

    #output:
    #xy    -  x/y or lon/lat of nodes
    #depth - depth value at node points
    #ect   - element connection table
    #bnd   - list of boundary nodes

    with open(filename, 'r') as f:
        # Skip mesh format lines
        for _ in range(4):  # Skip 4 lines directly (including $Nodes)
            next(f)
        
        # Read number of nodes
        nn = int(next(f).strip())
        
        # Initialize node coordinate and depth arrays
        xy = np.zeros((nn, 2), dtype=np.double)
        depth = np.zeros(nn, dtype=np.double)
        
        # Read coordinates and depths
        for i in range(nn):
            line = next(f).split()
            idx = int(line[0]) - 1
            x_coord = float(line[1])
            xy[idx, 0] = x_coord - 360 if x_coord > 180 else x_coord
            xy[idx, 1] = float(line[2])
            depth[idx] = float(line[3])
        
        # Skip '$EndNodes' and '$Elements'
        next(f)
        next(f)
        
        # Read number of elements
        ne = int(next(f).strip())
        
        # Initialize temporary arrays to read in element info
        ecttemp = np.zeros((ne, 3), dtype=np.int32)
        bndtemp = []
        
        elem_count = 0
        for _ in range(ne):
            line = next(f).split()
            eltype = int(line[1])
            if eltype == 15:
                bndtemp.append(int(line[5]) - 1)
            else:
                #ecttemp[elem_count, :] = [int(line[6]) - 1, int(line[7]) - 1, int(line[8]) - 1]
                ecttemp[elem_count, :] = [int(line[-3]) - 1, int(line[-2]) - 1, int(line[-1]) - 1]
                elem_count += 1
        
        # Trim the ect array to the actual number of elements
        ect = ecttemp[:elem_count, :]
        bnd = np.array(bndtemp, dtype=np.int32)
    
    return xy, depth, ect, bnd

def calc_elm_size(xy, ect):

    #purpose: Calculate element size, by calculating the distance between each node on the triangle, then
   # saving the minimum or maximum distance for each node

   #input:
   # xy  -  x/y or lon/lat of nodes
   # ect - element connection table

   #output:
   # distmin(number of nodes)  - the minimum distance between this node and any connected node point
   # distmax(number of nodes)  - the maximum distance between this node and any connected node point

    nn = len(xy)
    ne = len(ect)
    radiusofearth = 6378.137  # radius of earth at equator
    d2r = np.pi / 180  # degrees to radians conversion factor

    # Initialize arrays to store min and max distances
    distmin = np.full(nn, np.inf)
    distmax = np.zeros(nn)

    # Precompute cosines and sines for latitudes
    cos_lat = np.cos(xy[:, 1] * d2r)
    sin_lat = np.sin(xy[:, 1] * d2r)

    # Calculate distances for all elements
    for i, j, k in ect:
        # Extract coordinates
        lon_i, lon_j, lon_k = xy[i, 0], xy[j, 0], xy[k, 0]
        cos_lat_i, cos_lat_j, cos_lat_k = cos_lat[i], cos_lat[j], cos_lat[k]
        sin_lat_i, sin_lat_j, sin_lat_k = sin_lat[i], sin_lat[j], sin_lat[k]

        # Calculate distances using the Haversine formula
        dist_ij = 2 * radiusofearth * np.arcsin(np.sqrt(np.sin((xy[j, 1] - xy[i, 1]) * d2r / 2) ** 2 +
                                                       cos_lat_i * cos_lat_j * np.sin((lon_j - lon_i) * d2r / 2) ** 2))
        dist_jk = 2 * radiusofearth * np.arcsin(np.sqrt(np.sin((xy[k, 1] - xy[j, 1]) * d2r / 2) ** 2 +
                                                       cos_lat_j * cos_lat_k * np.sin((lon_k - lon_j) * d2r / 2) ** 2))
        dist_ki = 2 * radiusofearth * np.arcsin(np.sqrt(np.sin((xy[i, 1] - xy[k, 1]) * d2r / 2) ** 2 +
                                                       cos_lat_k * cos_lat_i * np.sin((lon_i - lon_k) * d2r / 2) ** 2))

        # Update min and max distances for each node
        distmin[i], distmax[i] = min(distmin[i], dist_ij, dist_ki), max(distmax[i], dist_ij, dist_ki)
        distmin[j], distmax[j] = min(distmin[j], dist_ij, dist_jk), max(distmax[j], dist_ij, dist_jk)
        distmin[k], distmax[k] = min(distmin[k], dist_jk, dist_ki), max(distmax[k], dist_jk, dist_ki)

    return distmin, distmax

def create_mask(xy, ect):
    # Extract x-coordinates of the nodes for each element
    x_coords = xy[ect, 0]  # ect indexes rows in xy, column 0 is x-coordinates

    # Check for cross-dateline elements
    # Calculate signs and check if all are the same for each element
    signs = np.sign(x_coords)
    uniform_sign = (np.ptp(signs, axis=1) == 0)  # Change in peak-to-peak (max-min) across rows; if 0, all signs are the same

    # Check for elements near the dateline, i.e., abs(x) < 10
    near_dateline = (np.abs(x_coords) < 10).any(axis=1)  # Check any coordinate < 10 in absolute value for each element

    # Combine conditions: elements are valid if they do not cross the dateline and are not near the dateline
    mask = ~(uniform_sign | near_dateline)

    return mask.astype(int)  # Convert boolean mask to integer (1s and 0s)



def setup_plot(ax, extent=[-180, 180, -90, 90], proj=ccrs.PlateCarree()):
    """ Common plot setup function """
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.COASTLINE, zorder=101)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    gl.xlocator = mticker.FixedLocator(np.linspace(extent[0], extent[1], 7))
    gl.ylocator = mticker.FixedLocator(np.linspace(extent[2], extent[3], 7))
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()

def plot_eleminfo(plotdescriptor, xy, ect, distmin, distmax, depth, highlight_nodes=None, extent=[-180, 180, -90, 90], plotDir='./', region='Global', station=[]):
    print('Grid')
    print(plotdescriptor)
    print('Min/max of distmin:', np.min(distmin), np.max(distmin))
    print('Min/max of distmax:', np.min(distmax), np.max(distmax))
    print('Min/max of bathy:', np.min(depth), np.max(depth))

  
    #create one tringulation and use it for multiple plots: 
    ## first check if region is Arctic
    if region.startswith('Arctic') or region.startswith('ARC'):
        proj = ccrs.NorthPolarStereo()
        x,y,_ = proj.transform_points(ccrs.PlateCarree(), xy[:,0], xy[:,1]).T
        pmask = np.invert(np.logical_or(np.isinf(x), np.isinf(y)))
        xy[:,0] = np.compress(pmask, x)
        xy[:,1] = np.compress(pmask, y)
        triang=tri.Triangulation(xy[:,0],xy[:,1],triangles=ect)#, mask=mask)
    else:
        if region.startswith('Svalbard'):
            proj = ccrs.Stereographic(central_latitude=75, central_longitude=30)
        else:
            ## next plot global on Robinson projection
            proj = ccrs.Robinson()
        x, y, _ = proj.transform_points(ccrs.PlateCarree(), xy[:,0], xy[:,1]).T
        pmask = np.invert(np.logical_or(np.isinf(x), np.isinf(y)))
        xy[:,0] = np.compress(pmask, x)
        xy[:,1] = np.compress(pmask, y)
        mask = create_mask(xy,ect)
        triang=tri.Triangulation(xy[:,0],xy[:,1],triangles=ect, mask=mask)
    if 'NWA' in plotdescriptor or 'ARC' in plotdescriptor:
        print(f"{plotdescriptor} is RDWPS region. Changing plot limits.")
        vpltmin = 1
        vpltmax = 5
    else:
        vpltmin=np.round(np.min(distmin))
        vpltmax= np.round(np.max(distmax))
    
    # Shared figure setup
    figsize = [18.0, 9.0]
#    plots = [
#        ('elm', 'b-', 'Element outlines', None, 'jet')
#        ]
    elm_shad = 'gouraud' # can be flat or auto
    elm_cmap = 'plasma_r'
    plots = [
        ('elm', 'b-', 'Element outlines', None, 'jet'),
        ('maxsize', distmax, 'Element size in km', elm_shad, elm_cmap),
        ('minsize', distmin, 'Element size in km', elm_shad, elm_cmap),
        ('bathy', depth, 'Bathymetry in m', None, 'jet')
    ]

    for suffix, data, label, shading, cmap in plots:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=proj)
        if not region.startswith('Arctic'): setup_plot(ax, extent=extent)
        if suffix == 'elm':
            cf = ax.triplot(triang, data, linewidth=0.1)
        else:
            cf = ax.tripcolor(triang, data, cmap=cmap, shading=shading if shading else 'flat', vmin=vpltmin, vmax=vpltmax if data is not depth else None)
            plt.colorbar(mappable=cf, label=label, fraction=0.036, pad=0.04)

#        if highlight_nodes is not None and suffix == 'elm':
#            ax.scatter(xy[highlight_nodes, 0], xy[highlight_nodes, 1], color='red', s=16, zorder=102)
        
        if region.startswith('Arctic'):
            polarCentral_set_latlim([extent[2],extent[3]], ax)
        add_map_features(ax)
        if len(station) > 0:
            ax.plot(station[0], station[1], 'rx', transform=ccrs.PlateCarree())

        fig.savefig(os.path.join(plotDir, f'{suffix}_{plotdescriptor}_{region}.png'))
        plt.close(fig)
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

#filename = "./uglo_poly_nBlkS.ww3"
#descriptor = "uglo_poly_nBlkS"
#descriptor = "25km_unstr_uniform"
descriptor = sys.argv[1]
region = sys.argv[2]
print(f"Looking at mesh {descriptor} in region {region}")
filename = f"./data/{descriptor}.ww3"
#filename = f"/home/gsu000/l5/GDWPS/unstruc_oceanmesh2d/{descriptor}.ww3"
#filename = f"/home/gsu000/l5/RDWPS/nwaunstr/{descriptor}.ww3"
doff = 0.5
plot_regions = {
        "EastCoast":[-70, -45, 42, 62],
        "Global":[-180, 180, -90, 90],
        "WestCoast": [-135, -122, 48, 55],
        "Arctic":[-180, 180, 75, 90],
        "ArcticPole":[-180,180,88,90],
        "NoPoleArctic":[150,180,80,89.5],
        "GSL":[-70, -55, 45, 52],
        "GulfOfMexico":[-98,-83,19,30],
        "NWA":[-98,-38, 25, 70],
        "ARC":[-180, 180, 50, 90],
        "Svalbard":[-20, 30, 75, 82],
        "Ausuittuq":[-82.896-doff, -82.896+doff, 76.416-doff, 76.416+doff],
        "Ikaluktutiak":[-115.062-doff, -115.062+doff, 67.846-doff, 67.846+doff],
        "Kugluktuk":[-105.634-doff, -105.634+doff, 69.166-doff, 69.166+doff]
        }

plot_stations = {
        "Ausuittuq":[-82.896, 76.416],
        "Ikaluktutiak":[-115.062, 67.846],
        "Kugluktuk":[-105.634, 69.166]
        }


xy, depth, ect, bnd =  read_gmsh(filename)


## maximum latitude
if region.startswith('Arctic'):
    max_lat = xy[:,-1].max()
    inod = np.argmax(xy[:,1])
    nod = inod + 1
    #max_lat_cos = np.cos(max_lat*np.pi/180)
    print(f'Max lat is {max_lat:.5f} correpsonding to node {nod}')
    ## find elements with one node as max
    itot = np.where(ect==inod)
    highlight_elms = itot[0]
    highlighted_nodes = []
    elem_lat_max = -90.0
    for el in highlight_elms:
        latm, lonm = 0.0, 0.0
        for n in ect[el,:]:
            lonm += xy[n,0]
            latm += xy[n,1]
            if n not in highlighted_nodes:
                highlighted_nodes.append(n)
        lonm /= 3
        latm /= 3
        if latm > elem_lat_max:
            elem_max = el
            elem_lat_max = latm
            elem_lon_max = lonm
        print(f"Element: {el}")
        print(f"Nodes: {ect[el,0]+1}, {ect[el,1]+1}, {ect[el,2]+1} : Mean Lon/Lat {lonm:.3f}/{latm:.3f}")
    print('Lon/Lat of highlighted nodes are:')
    for n in highlighted_nodes:
        print(f' Node {n+1}: {xy[n,0]:9.3f} E, {xy[n,1]:.3f} N')
    print(f"Maximum element is {elem_max} with cell lat of {elem_lat_max:.3f}")
else:
    highlighted_nodes = []


distmin, distmax = calc_elm_size(xy, ect)
#highlighted_nodes = [inod]# Replace with your actual node indices , 12776, 13923
extent = plot_regions[region]
if region in plot_stations.keys():
    station = plot_stations[region]
else:
    station = []
plotDir = f'/home/gsu000/public_html/GDWPS/unstruc/mesh'
if not os.path.exists(plotDir): os.makedirs(plotDir)
plot_eleminfo(descriptor, xy, ect, distmin, distmax, depth, highlight_nodes=highlighted_nodes, extent=extent, plotDir=plotDir, region=region, station=station)
