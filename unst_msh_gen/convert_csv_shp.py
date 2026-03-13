import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import sys, os

inFile = 'boundary_coords_nep_closed.csv'
outFile = 'boundary_coords_nep.shp'

df = pd.read_csv(inFile, names=['Longitude','Latitude'])
# convert to +- 180
df['Longitude'].loc[df['Longitude']>180] -= 360

poly_geom = Polygon(zip(df.Longitude, df.Latitude))
poly = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[poly_geom])

poly.to_file(outFile, driver='ESRI Shapefile')
