import pygmt
import xarray as xr
import numpy as np

# 1. Set region and resolution
# 'l' = low, 'i' = intermediate, 'h' = high, 'f' = full
region_label = "great_lakes"
#region = [-180, 180, -90, 90] # Global
region = [-93, -75, 40, 50] # great_lakes
spacing = "5s"          # 0.1 degree resolution d=degrees, m=minutes, s=seconds, e=metre, f=foot, k=km, n=nautical mile
res = 'f'                 # Intermediate resolution

# 0 = Ocean, 1 = Land
mask_values=[0, 1, 0, 1, 0] # [ocean, land, lake, island, pond]
# 2. Use pygmt to create a land/sea mask grid from GSHHS
grid = pygmt.grdlandmask(
    region=region,
    spacing=spacing,
    resolution=res,
    mask_values=mask_values,
)

# 3. Convert pygmt grid to xarray dataset
ds = grid.to_dataset(name="landmask")

# 4. Add metadata (CF conventions)
ds.attrs['description'] = 'GSHHS Land/Sea Mask'
ds.landmask.attrs['units'] = 'flag'
ds.landmask.attrs['long_name'] = 'land_sea_mask'

# 5. Save as NetCDF
output_filename = f"./data/gshhs_mask_{res}_{spacing}_{region_label}.nc"
ds.to_netcdf(output_filename)

print(f"Raster saved to {output_filename}")

