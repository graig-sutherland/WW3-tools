""" Jigsaw meshes for WW3 with global bathymetry
"""

# Authors: Ali Salimi-Tarazouj, Darren Engwirda

# This script will create mesh spacing based on user defined windows in json format or based on polygons in shapefile format

import configparser
import numpy as np
import netCDF4 as nc
import argparse
import json
import geopandas as gpd
from shapely.geometry import Point
from spacing import *
import os

def great_circle(R, lon1, lat1, lon2, lat2):
    '''
    returns great circle distance using spherical law of cosines
    '''
    dlon = lon2 - lon1
    distance = R * np.arccos(np.sin(lat1)*np.sin(lat2) + np.cos(lat1)*np.cos(lat2)*np.cos(dlon))
    return distance

def parse_input_args():
    parser = argparse.ArgumentParser(description='Create a mask file with multiple methods.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    args = parser.parse_args()
    return args

def load_configuration(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    mask_file = config.get('CommandLineArgs', 'mask_file', fallback='wmask.nc')
    hmax = float(config.get('Spacing', 'hmax', fallback='100.0'))
    hmin = float(config.get('Spacing', 'hmin', fallback='100.0'))
    nwav = int(config.get('Spacing', 'nwav', fallback='-400'))
    hcenter = float(config.get('Spacing', 'hcenter', fallback='-100'))
    hbeta = float(config.get('Spacing', 'hbeta', fallback='-100'))
    arctic_hmax_lat = float(config.get('MeshSettings', 'arctic_hmax_lat', fallback='90'))
    arctic_hmax_val = float(config.get('MeshSettings', 'arctic_hmax_val', fallback='100'))

    try:
        windows = json.loads(config['DataFiles']['window_file']) if 'window_file' in config['DataFiles'] else None
    except json.JSONDecodeError:
        print("Error parsing windows data")
        breakpoint()
        windows = None

    try:
        shapefiles = json.loads(config['DataFiles']['shape_file']) if 'shape_file' in config['DataFiles'] else None
    except json.JSONDecodeError:
        print("Error parsing shapefiles data")
        shapefiles = None
        breakpoint()
    
    try:
        gaussian = json.loads(config['DataFiles']['gaussian_file']) if 'gaussian_file' in config['DataFiles'] else None
    except json.JSONDecodeError:
        print("Error parsing gaussian data")
        gaussian = None

    if shapefiles:
        shapefiles = [(shp['path'], shp['scale']) for shp in shapefiles]

    dem_file = config['DataFiles']['dem_file'] if 'dem_file' in config['DataFiles'] else None


    return windows, shapefiles, dem_file, gaussian, mask_file, hmax, hmin, nwav, arctic_hmax_lat, arctic_hmax_val, hcenter, hbeta


def create_mask_file(data_filename, output_filename, windows=None, shapefiles=None, gaussian=None, hmax=25, hmin=25, nwav=-400, arctic_hmax_lat=90., arctic_hmax_val=25, hcenter=-100, hbeta=0.1, gshhs=False):
    # Load DEM data
    data = nc.Dataset(data_filename, "r")
    xlon = np.asarray(data["lon"][:])
    ylat = np.asarray(data["lat"][:])
    elev = np.asarray(data["bed_elevation"][:], dtype=np.float32) #+ np.asarray(data["ice_thickness"][:], dtype=np.float32)

    # Compute midpoints for longitude and latitude
    xmid = 0.5 * (xlon[:-1] + xlon[1:])
    ymid = 0.5 * (ylat[:-1] + ylat[1:])
    xmat, ymat = np.meshgrid(xmid, ymid)

    land = form_land_mask_connect(elev, edry=2) >= 1
    high = form_land_mask_connect(elev, edry=8) >= 1

#-- init. h(x) data: impose global "reachable" land mask

    scal = np.full(
        (elev.shape[:]), hmax)
        
    scal[land] = hmax
    # apply SWE scaling
    if (nwav > 0) and (hcenter > 0):
        import sys; sys.exit("Can not have both nwav > 0 and hcenter > 0")
    else:
        if (nwav > 0.0):
            print(f"Calculating depth dependent size for nwave = {nwav}")
            scal = np.minimum(
                scal, swe_wavelength_spacing(
                    elev, land, nwav, hmin, hmax))
        if (hcenter > 0.0):
            print(f"Calculating depth dependent size for tanh centered at {hcenter} and stretching {hbeta}")
            scal = np.minimum(
                scal, tanh_wavelength_spacing(
                    elev, hmin, hmax, hcenter, hbeta))
    # check if I want to change the sizing after the mask_file is applied
    if arctic_hmax_lat < 90:
        print(f"create_siz: Applying scale of {arctic_hmax_val} km for lat > {arctic_hmax_lat:.1f}")
        ymid = 0.5*(ylat[:-1]+ylat[1:])
        scal[ymid>arctic_hmax_lat,:] = arctic_hmax_val
    else:
        print(f"create_siz: No Arctic max val and using prescribed scal val")

#-- final h(x) data: impose global "shoreline" min. val.
   
    scal[high] = hmax
   
    scal = setup_shoreline_pixels(scal, land, hmin)

    # apply NWA region to be max outside bounding box
    apply_region = True
    if apply_region:
        rasterDir = '/home/gsu000/data/ppp7/GEBCO'
        glon = np.genfromtxt(os.path.join(rasterDir, 'nwa5km.lon'))
        glat = np.genfromtxt(os.path.join(rasterDir, 'nwa5km.lat'))
        # convert glont to +-180
        glon = np.mod(glon+180, 360) - 180
        # Subset the meshgrid
        mask_x = (xmid >= glon.min()) & (xmid <= glon.max())
        mask_y = (ymid >= glat.min()) & (ymid <= glat.max())
        # going to apply hmax to points outside
        scal[np.ix_(~mask_y, ~mask_x)] = hmax

    # apply gaussian spacing
    if gaussian:
        for gauss in gaussian:
            R, sig = gauss['radius'], gauss['sigma']
            hmax, hmin = gauss['hmax'], gauss['hmin']
            lon0, lat0 = np.deg2rad(gauss['lon0']), np.deg2rad(gauss['lat0'])
            dist = great_circle(R, np.deg2rad(xmat), np.deg2rad(ymat), lon0, lat0)
            dh = hmax - hmin
            dr = dist/sig
            hfun = hmax - dh * np.exp(-0.5 * dr**2)
            # only apply over 3 sigma 
            scal[dist<2*sig] = hfun[dist<2*sig]

    # Apply window-based refinement if provided
    if windows:
        print(f"Applying window functions. Using tanh spacing")
        hmin_all_windows = hmax
        for nwin, window in enumerate(windows):
            print(f"window {nwin}: ll, ur = ({window['min_lon']:.1f}, {window['min_lat']:.1f}), ({window['max_lon']:.1f}, {window['max_lat']:.1f}")
            if window["hmin"] < hmin_all_windows: hmin_all_windows = window["hmin"]
            # Subset the meshgrid
            mask_x = (xmid >= window['min_lon']) & (xmid <= window['max_lon'])
            mask_y = (ymid >= window['min_lat']) & (ymid <= window['max_lat'])
            # going to apply swe scaling to subset
            #ewin = elev[window_mask]
            ewin = elev[np.ix_(mask_y,mask_x)]
            valswin = scal[np.ix_(mask_y, mask_x)]
            # going to apply scaling
            valswin = np.minimum(valswin, tanh_wavelength_spacing(ewin,window["hmin"],window["hmax"],hcenter,hbeta))
            scal[np.ix_(mask_y, mask_x)] = valswin
        
        print(f"Apply {hmin_all_windows} km resolution to coastline of windows")
        scal[high] = hmax
        scal = setup_shoreline_pixels(scal, land, hmin_all_windows)

    # Process shapefiles if provided
    if shapefiles:
        default_scale = hmax
        for shapefile, scale in shapefiles:
            print(f"Applying {scale} km spacing to {shapefile}")
            gdf = gpd.read_file(shapefile)
            minx, miny, maxx, maxy = gdf.total_bounds
            # Subset the meshgrid
            mask_x = (xmid >= minx) & (xmid <= maxx)
            mask_y = (ymid >= miny) & (ymid <= maxy)
            x_sub, y_sub, e_sub, vals_sub = xmid[mask_x], ymid[mask_y], elev[np.ix_(mask_y,mask_x)], scal[np.ix_(mask_y, mask_x)]

            # apply tanh scaling to subset then only pick the points in polygon
            vals_sub = np.minimum(vals_sub, tanh_wavelength_spacing(e_sub, scale, default_scale, hcenter, hbeta))
            # output scale
            scal[np.ix_(mask_y, mask_x)] = vals_sub

#            xmat_sub, ymat_sub = np.meshgrid(x_sub, y_sub)
#
#            # Create GeoDataFrame for the points in the subset
#            points_sub = [Point(x, y) for x, y in zip(xmat_sub.flatten(), ymat_sub.flatten())]
#            points_gdf_sub = gpd.GeoDataFrame(geometry=points_sub)
#            points_gdf_sub.set_crs(gdf.crs, inplace=True)
#
#            # Spatial join - ideally I don't have the default_scale if not within sub bbox
#            #points_within = gpd.sjoin(points_gdf_sub, gdf, how='inner', predicate='within')
#            #breakpoint()
#
#            # Spatial join only for the subset area
#            overlay_sub = gpd.sjoin(points_gdf_sub, gdf, how="left", predicate='intersects')
#            breakpoint()
#            overlay_sub['scale'] = overlay_sub['index_right'].apply(lambda x: scale if x >= 0 else default_scale)
#            scales_sub = overlay_sub['scale'].to_numpy().reshape(xmat_sub.shape)
#
#            # Map the scales back to the original full matrix
#            x_indices = np.searchsorted(xmid, x_sub)
#            y_indices = np.searchsorted(ymid, y_sub)
#            scal[np.ix_(mask_y, mask_x)] = scales_sub

    
    # Write results to a NetCDF file
    data_out = nc.Dataset(output_filename, "w")
    data_out.createDimension("nlon", xmid.size)
    data_out.createDimension("nlat", ymid.size)
    var = data_out.createVariable("val", "f4", ("nlat", "nlon"))
    lon = data_out.createVariable("lon", "f4", ("nlon"))
    lat = data_out.createVariable("lat", "f4", ("nlat"))
    var[:, :] = scal
    lon[:] = xmid
    lat[:] = ymid
    data_out.close()

if __name__ == "__main__":
    args = parse_input_args()
    windows, shapefiles, dem_file, gaussian, mask_file, hmax, hmin, nwav, arctic_hmax_lat, arctic_hmax_val, hcenter, hbeta = load_configuration(args.config)
    create_mask_file(dem_file, mask_file, windows, shapefiles, gaussian, hmax, hmin, nwav, arctic_hmax_lat, arctic_hmax_val, hcenter, hbeta)

