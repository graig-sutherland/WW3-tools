import configparser
import argparse
import numpy as np
import geopandas as gpd
import jigsawpy
import os, sys

def parse_input_args():
    parser = argparse.ArgumentParser(description='Create a mask file with multiple methods.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    return args

def load_configuration(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    bbox_file = config['DataFiles']['bbox_file'] if 'bbox_file' in config['DataFiles'] else None
    geom_file = config['DataFiles']['geom_file'] if 'geom_file' in config['DataFiles'] else None

    return bbox_file, geom_file

if __name__ == "__main__":
    args = parse_input_args()
    bbox_file, geom_file = load_configuration(args.config)

    if bbox_file is not None and geom_file is not None:

        bbox_lon, bbox_lat = np.loadtxt(bbox_file, delimiter=',', unpack=True)
        
        # write to msh_t format
        geom = jigsawpy.jigsaw_msh_t()
        geom.mshID = "euclidean-mesh"
        geom.ndims = +2
        geom.radii = np.full(
            3, 6.371E+003, dtype=geom.REALS_t)
        geom.vert2 = np.array([((lo,la),0) for lo,la in zip(bbox_lon,bbox_lat)], dtype=geom.VERT2_t)
        geom.edge2 = np.array([((ii,ii+1),0) for ii in range(len(bbox_lon))], dtype=geom.EDGE2_t)
        # fix last index
        geom.edge2['index'][-1,-1] = int(0)
    
        jigsawpy.savemsh(geom_file, geom)

#    ## read in coastline data
#    coastlineDir = '/home/gsu000/data/ppp8/CoastlineData/GSHHS_shp'
#    res = 'h'
#    gshhsFile = os.path.join(coastlineDir, res, f'GSHHS_{res}_L1.shp')
#    coast_gdf = gpd.read_file(gshhsFile)
#    # just land/ocean boundaries
#    coast_gdf_oce = coast_gdf[coast_gdf['level'] == 1]
#    # make geometry valid
#    coast_gdf_oce['geometry'] = coast_gdf_oce['geometry'].make_valid()
#
#    ## name region
#    #region = "NWA"
#    region = sys.argv[1]
#
#    # define base geoseries. I have coordinates from Patrick
#    rasterDir = '/home/gsu000/data/ppp7/GDWPS'
#    # lets try to read bndy file
#    bbox_lon, bbox_lat = np.loadtxt(os.path.join(rasterDir, f"{region.lower()}5km.bndy"), delimiter=',', unpack=True)
#    
#    # write to msh_t format
#    geom = jigsawpy.jigsaw_msh_t()
#    geom.mshID = "euclidean-mesh"
#    geom.ndims = +2
#    geom.radii = np.full(
#        3, 6.371E+003, dtype=geom.REALS_t)
##    geom.vert2 = np.array([((pt[0],pt[1]),0) for pt in all_points], dtype=geom.VERT2_t)
##    geom.edge2 = np.array([((ia,ib),0) for ia, ib in edges], dtype=geom.EDGE2_t)
#    geom.vert2 = np.array([((lo,la),0) for lo,la in zip(bbox_lon,bbox_lat)], dtype=geom.VERT2_t)
#    geom.edge2 = np.array([((ii,ii+1),0) for ii in range(len(bbox_lon))], dtype=geom.EDGE2_t)
#    # fix last index
#    geom.edge2['index'][-1,-1] = int(0)
#
#    jigsawpy.savemsh(f"/home/gsu000/data/ppp7/RDWPS/{region.upper()}_geom_bbox.msh", geom)
