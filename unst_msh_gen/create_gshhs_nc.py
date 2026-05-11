import pygmt
import xarray as xr
import numpy as np

# 1. Set region and resolution
# 'l' = low, 'i' = intermediate, 'h' = high, 'f' = full
region = [-180, 180, -90, 90] # Global
spacing = "1m"          # 0.1 degree resolution d=degrees, m=minutes, s=seconds, e=metre, f=foot, k=km, n=nautical mile
res = 'f'                 # Intermediate resolution

# 2. Use pygmt to create a land/sea mask grid from GSHHS
# 0 = Ocean, 1 = Land
grid = pygmt.grdlandmask(
    region=region,
    spacing=spacing,
    resolution=res,
    mask_values=[0, 1]#, 1, 1, 1], # [ocean, land, lake, island, pond]
)

# 3. Convert pygmt grid to xarray dataset
ds = grid.to_dataset(name="landmask")

# 4. Add metadata (CF conventions)
ds.attrs['description'] = 'GSHHS Land/Sea Mask'
ds.landmask.attrs['units'] = 'flag'
ds.landmask.attrs['long_name'] = 'land_sea_mask'

# 5. Save as NetCDF
output_filename = f"./data/gshhs_mask_{res}_{spacing}.nc"
ds.to_netcdf(output_filename)

print(f"Raster saved to {output_filename}")

