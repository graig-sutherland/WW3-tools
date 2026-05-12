import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Polygon, MultiPolygon
import jigsawpy
import os, sys
import matplotlib.pyplot as plt
        
def simplify_boundaries(boundaries, tolerance):
    simplified = []
    for coords in boundaries:
        line = LineString(coords)
        simp_line = line.simplify(tolerance, preserve_topology=True)
        simplified.append(list(simp_line.coords))
    return simplified

def extract_rings(geom):
    '''Yield lists of (lon, lat) for all exteriors of Polygons.'''
    if geom.geom_type == 'Polygon':
        yield list(geom.exterior.coords)
    elif geom.geom_type == 'MultiPolygon':
        for part in geom.geoms:
            yield list(part.exterior.coords)

if __name__ == "__main__":
    ## read in coastline data
    coastlineDir = '/home/gsu000/data/ppp8/CoastlineData/GSHHS_shp'
    res = 'h'
    gshhsFile = os.path.join(coastlineDir, res, f'GSHHS_{res}_L1.shp')
    coast_gdf = gpd.read_file(gshhsFile)
    # just land/ocean boundaries
    coast_gdf_oce = coast_gdf[coast_gdf['level'] == 1]
    # make geometry valid
    coast_gdf_oce['geometry'] = coast_gdf_oce['geometry'].make_valid()

    # define base geoseries. I have coordinates from Patrick
    rasterDir = '/home/gsu000/data/ppp7/GEBCO'
    glon = np.genfromtxt(os.path.join(rasterDir, 'nwa5km.lon'))
    glat = np.genfromtxt(os.path.join(rasterDir, 'nwa5km.lat'))
    bbox_lon = np.concatenate((glon[0,::-1], glon[:,0].T, glon[-1,:], glon[::-1,-1]))
    bbox_lon = np.mod(bbox_lon+180, 360) - 180
    bbox_lat = np.concatenate((glat[0,::-1], glat[:,0].T, glat[-1,:], glat[::-1,-1]))
    
    # write to msh_t format
    geom = jigsawpy.jigsaw_msh_t()
    geom.mshID = "euclidean-mesh"
    geom.ndims = +2
    geom.radii = np.full(
        3, 6.371E+003, dtype=geom.REALS_t)
#    geom.vert2 = np.array([((pt[0],pt[1]),0) for pt in all_points], dtype=geom.VERT2_t)
#    geom.edge2 = np.array([((ia,ib),0) for ia, ib in edges], dtype=geom.EDGE2_t)
    geom.vert2 = np.array([((lo,la),0) for lo,la in zip(bbox_lon,bbox_lat)], dtype=geom.VERT2_t)
    geom.edge2 = np.array([((ii,ii+1),0) for ii in range(len(bbox_lon))], dtype=geom.EDGE2_t)
    # fix last index
    geom.edge2['index'][-1,-1] = int(0)

    jigsawpy.savemsh(f"NWA_geom_outside.msh", geom)
    import sys; sys.exit()
    ## not running below code at the moment

    boundary_coords = np.vstack((bbox_lon,bbox_lat)).T
    # convert to geopandas series
    search_area = Polygon(boundary_coords)
    bound_gdf = gpd.GeoDataFrame(index=[0], crs=coast_gdf_oce.crs, geometry=[search_area])

    # first I'm going to clip
    coast_in_region = gpd.clip(coast_gdf_oce, bound_gdf)
    # get new geomeotery
    mesh_domain = pd.concat([bound_gdf, coast_in_region], ignore_index=True)
    
    # plot
    mesh_domain.boundary.plot()
    plt.savefig(f"/home/gsu000/public_html/geopandas_boundary.png")
    plt.clf()
    # extract boundaries as point lists
    boundary_loops = []
    for geom in mesh_domain.geometry:
        for coords in extract_rings(geom):
            boundary_loops.append(coords)

    # get mesh data
    all_points = []
    edges = []
    offset = 0
    simplifyLoop = False
    if simplifyLoop:
        tolerance = 1.0/12
        filtered_loops = simplify_boundaries(boundary_loops, tolerance)
        simplify_ext = f"simplify_tol_{tolerance:.2f}"
    else:
        filtered_loops = boundary_loops
        simplify_ext = f"nosimplify"

    for loop in boundary_loops:
        n = len(loop)
        all_points.extend(loop)
        # Closed loop: edge from each point to next, wrapping around
        edges += [(offset + i, offset + (i + 1) % n) for i in range(n)]
        offset += n

    # write to msh_t format
    geom = jigsawpy.jigsaw_msh_t()
    geom.mshID = "euclidean-mesh"
    geom.ndims = +2
    geom.radii = np.full(
        3, 6.371E+003, dtype=geom.REALS_t)
    geom.vert2 = np.array([((pt[0],pt[1]),0) for pt in all_points], dtype=geom.VERT2_t)
    geom.edge2 = np.array([((ia,ib),0) for ia, ib in edges], dtype=geom.EDGE2_t)

    jigsawpy.savemsh(f"NWA_geom_outside.msh", geom)
