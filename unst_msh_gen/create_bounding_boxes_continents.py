import geopandas as gpd
import numpy as np
import shapely.geometry as sg
import os

#
## Step 1: Load Natural Earth 10m Bathymetry for the -200m depth contour
## You can download 'ne_10m_bathymetry_K_200.shp' from Natural Earth
#bathyDir = "/home/gsu000/data/ppp7/NaturalEarth/bathy"
#shelf_edge = gpd.read_file(os.path.join(bathyDir, 'ne_10m_bathymetry_K_200.shp'))
#
## Step 2: Load country boundaries to isolate and omit Canada
##world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
#worldFile = "/home/gsu000/data/ppp7/NaturalEarth/cultural/ne_110m_admin_0_countries.shp"
#world = gpd.read_file(worldFile)
#canada_geom = world[world['SOVEREIGNT'] == 'Canada'].geometry.union_all()
#
## Step 3: Extract and filter the continental shelf line strings
## We combine the shelf vectors and subtract Canada's landmass/EEZ zone
#shelf_lines = shelf_edge.geometry.union_all()
#non_canadian_shelf = shelf_lines.difference(canada_geom.buffer(0.1))
#
## Step 4: Define 1/12 degree resolution parameters
#step_size = 1.0 / 4.0
#lon_grid = np.arange(-180.0, 180.0, step_size)
#lat_grid = np.arange(-90.0, 90.0, step_size)
#
#bounding_boxes = []
#
## Step 5: Generate the bounding boxes intersecting the continental shelf
#bounds = non_canadian_shelf.bounds  # [min_lon, min_lat, max_lon, max_lat]
#
## Filter global grids to match the total active geospatial footprint
#local_lons = lon_grid[(lon_grid >= bounds[0] - step_size) & (lon_grid <= bounds[2])]
#local_lats = lat_grid[(lat_grid >= bounds[1] - step_size) & (lat_grid <= bounds[3])]
#print('first break ...')
#breakpoint()
#for lat in local_lats:
#    for lon in local_lons:
#        box = sg.box(lon, lat, lon + step_size, lat + step_size)
#        
#        # Keep the box if it intersects the -200m continental shelf slope
#        if non_canadian_shelf.intersects(box):
#            bounding_boxes.append(box.bounds)
#
#print(f"Generated {len(bounding_boxes)} shelf-extended bounding boxes.")
#breakpoint()

# World coastlines - Canada
import geopandas as gpd
import numpy as np
import shapely.geometry as sg

# Step 1: Load Natural Earth country polygons
#world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
worldFile = "/home/gsu000/data/ppp7/NaturalEarth/cultural/ne_110m_admin_0_countries.shp"
world = gpd.read_file(worldFile)

# Step 2: Isolate Canada's geometry to use as a mask
canada_geom = world[world['SOVEREIGNT'] == 'Canada'].geometry.union_all()

# Step 3: Filter out islands (Keep major continental landmasses)
# A threshold of 5.0 square degrees isolates the main continents
continents = world[world['geometry'].area > 3.0]

bounding_boxes = []
step_size = 1.0 / 4.0  # 0.08333333 degrees

lon_grid = np.arange(-180.0, 180.0, step_size)
lat_grid = np.arange(-90.0, 90.0, step_size)

# Step 4: Process continental coastlines with Canada omitted
for _, continent in continents.iterrows():
    # Extract the boundary line of the continent
    coastline = continent.geometry.boundary
    
    # Remove any portion of the coastline that touches or falls inside Canada
    # We buffer by a tiny fraction to ensure shared border lines are cleanly removed
    non_canadian_coastline = coastline.difference(canada_geom.buffer(0.001))
    
    if non_canadian_coastline.is_empty:
        continue
        
    bounds = non_canadian_coastline.bounds  # [min_lon, min_lat, max_lon, max_lat]
    
    # Filter the global grid arrays to match the local continental footprint
    local_lons = lon_grid[(lon_grid >= bounds[0] - step_size) & (lon_grid <= bounds[2])]
    local_lats = lat_grid[(lat_grid >= bounds[1] - step_size) & (lat_grid <= bounds[3])]
    
    for lat in local_lats:
        for lon in local_lons:
            box = sg.box(lon, lat, lon + step_size, lat + step_size)
            
            # Step 5: Keep the box only if it intersects the filtered coastline
            if non_canadian_coastline.intersects(box):
                bounding_boxes.append(box.bounds)

print(f"Generated {len(bounding_boxes)} non-Canadian continental bounding boxes.")
breakpoint()

## World coastlines
#import geopandas as gpd
#import numpy as np
#import shapely.geometry as sg
#
## Step 1: Load Natural Earth low-res land polygons
## Using naturalearth_lowres as a reliable baseline for continental filtering
#world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
#
## Step 2: Filter out all islands
## Keeping only the 6 primary continental landmasses by setting a large area threshold
#continents = world[world['geometry'].area > 3.0] 
#coastlines = continents.boundary
#
## Step 3: Define 1/12 degree resolution parameters
#step_size = 1.0 / 12.0  # Exactly 0.08333333 degrees
#
## Create a global grid aligned perfectly to integer intervals
#lon_grid = np.arange(-180.0, 180.0, step_size)
#lat_grid = np.arange(-90.0, 90.0, step_size)
#
#bounding_boxes = []
#
## Step 4: Spatial index grid optimization to keep processing fast
#for _, line in coastlines.items():
#    bounds = line.bounds # [min_lon, min_lat, max_lon, max_lat]
#    
#    # Filter the global grid arrays to only look within the continent's extent
#    local_lons = lon_grid[(lon_grid >= bounds[0] - step_size) & (lon_grid <= bounds[2])]
#    local_lats = lat_grid[(lat_grid >= bounds[1] - step_size) & (lat_grid <= bounds[3])]
#    
#    for lat in local_lats:
#        for lon in local_lons:
#            # Construct the 1/12 degree bounding box
#            box = sg.box(lon, lat, lon + step_size, lat + step_size)
#            
#            # Step 5: Keep the box only if it intersects a continental coastline
#            if line.intersects(box):
#                # Returns coordinates in standard [min_lon, min_lat, max_lon, max_lat] format
#                bounding_boxes.append(box.bounds)
#
#print(f"Generated {len(bounding_boxes)} continental bounding boxes at 1/12 degree resolution.")
#breakpoint()
