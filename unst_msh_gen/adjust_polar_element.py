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
    dlat = np.pi/2 - lat
    k = 2*R / (1.0 + np.sin(lat0)*np.sin(dat) + np.cos(lat0)*np.cos(dlat)*np.cos(dlon))
    # output
    x = k*np.cos(lat)*np.sin(dlon)
    y = k*np.cos(lat0)*np.sin(dlat) - k*np.sin(lat0)*np.cos(dlat)*np.cos(dlon)
    return x, y

def xy2ll(x, y, R=6378.0, lon0=0.0, lat0=np.pi/2):
    p = np.sqrt(x*x + y*y)
    c = 2.0 * np.arctan2(2.0*R, p)
    #output
    lat = np.arcsin(np.cos(c)*np.sin(lat0) + y*np.sin(c)*np.cos(lat0)/p)
    dlon = np.arctan2(p*np.cos(lat0)*np.cos(c) - y*np.sin(lat0)*np.sin(c), x*np.sin(c))
    lon = lon0 + dlon
    return lon, lat

def ll2xy_ps(lon, lat, R=6378.0):
    k0 = 1.0
    x = 2*R*k0*np.tan(np.pi/4-lat/2)*np.sin(lon)
    y = -2*R*np.tan(np.pi/4 - lat/2)*np.cos(lon)
    return x, y
def xy2ll_ps(x, y, R=6378.0):
    p = np.sqrt(x*x + y*y)
    c = 2.0 * np.arctan2(2.0*R, p)
    lon = np.arctan2(-y, x)
    lat = np.arcsin(np.cos(c))
    return lon, lat

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
                ecttemp[elem_count, :] = [int(line[6]) - 1, int(line[7]) - 1, int(line[8]) - 1]
                elem_count += 1
        
        # Trim the ect array to the actual number of elements
        ect = ecttemp[:elem_count, :]
        bnd = np.array(bndtemp, dtype=np.int32)
    
    return xy, depth, ect, bnd


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


if __name__ == "__main__":
    descriptor = sys.argv[1]
    print(f"Adjusting polar element for mesh {descriptor}")
    filename = f"./data/{descriptor}.ww3"
    
    ## read in mesh
    xy, depth, ect, bnd =  read_gmsh(filename)
    
    R = 6378.0 # radius of earth in km
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
    
    ## going to adjust
    print("going to move the northern most element to be centred on pole")
    rlon, rlat = xy[:,0]*np.pi/180, xy[:,1]*np.pi/180
    xps, yps, scl = stereo3(R, rlon, rlat, 0.0, np.pi/2, "fwd")
    #xps, yps = ll2xy_ps(rlon, rlat)
    # print xy of elem_max
    xm, ym = 0.0, 0.0
    node_max = []
    cnt = 0.0
    for n in ect[elem_max,:]:
        node_max.append(n)
        print(f"Node {n+1}: x = {xps[n]:9.3f} km, ys = {yps[n]:9.3f} km")
        xm += xps[n]
        ym += yps[n]
        cnt += 1
    # divide by 3
    xm /= cnt
    ym /= cnt
    # shift all highlighted_nodes
    print(f"Mean northern element is x = {xm:9.3f} km, y = {ym:9.3f} km")
    xs, ys, zs = np.copy(xps), np.copy(yps), np.copy(depth)
    xs[highlighted_nodes] -= 0.5*xm
    ys[highlighted_nodes] -= 0.5*ym
    xs[node_max] -= 0.5*xm
    ys[node_max] -= 0.5*ym
    # shift back to latlon
    lons, lats, iscl = stereo3(R, xs, ys, 0.0, np.pi/2, "inv")
    #lons, lats = xy2ll_ps(xs, ys)
    lons *= 180/np.pi
    lats *= 180/np.pi
    ## check if xs node_max
    for n in node_max:
        print(f"node {n+1}:   x = {xs[n]:9.3f} km,   y = {ys[n]:9.3f} km")
        print(f"node {n+1}: lat = {xs[n]:9.3f} km, lon = {ys[n]:9.3f} km")
        
    # look at nodes with lat > 88
    inord = np.where(xy[:,1]>88)
    # I need to interpolate depth to new nodes
    ffun = LinearNDInterpolator(
        list(zip(xps[inord], yps[inord])), depth[inord]
        )
    zs[highlighted_nodes] = ffun((xs[highlighted_nodes],ys[highlighted_nodes]))
    for n in highlighted_nodes:
        print(f' Node {n+1}: {xy[n,0]:9.3f} E, {xy[n,1]:.3f} N')
        print(f' Shifted   : {lons[n]:9.3f} E, {lats[n]:.3f} N')
    
    outFile = f"./data/{descriptor}_adjust.ww3"
    num_nodes = len(xy)
    xys = np.copy(xy)
    tri = np.copy(ect)
    tri += 1
    xys[highlighted_nodes,0] = lons[highlighted_nodes]
    xys[highlighted_nodes,1] = lats[highlighted_nodes]
    nodes = np.zeros((num_nodes,3))
    nodes[:,:2] = xys
    nodes[:,-1] = zs
    write_gmsh_mesh(outFile, nodes, tri)
