import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
import os

print(f"Reading in netcdf ...")
ds = xr.open_dataset("./data/spac.nc")
#ds = xr.open_dataset("./data/wmask_gauss.nc")
print(f"Start plotting ...")
skip = 4
lon = ds.lon.values[::skip]
lat = ds.lat.values[::skip]
spac = ds.val.values[::skip, ::skip]

fig, axis = plt.subplots(1, 1, subplot_kw=dict(projection=ccrs.Orthographic(-97,70)))

pcm = plt.contourf(lon, lat, spac, transform = ccrs.PlateCarree())
plt.colorbar(pcm, ax=axis)
#ds.val.plot(
#        ax=axis,
#        transform=ccrs.PlateCarree(),
#        cbar_kwargs={"orientation": "horizontal", "shrink":0.7},
#        robust=True,
#        )
axis.set_global()
axis.coastlines()
fig.savefig('/home/gsu000/public_html/GDWPS/spac_proc.pdf', bbox_inches='tight')
plt.close(fig)
print(f"Done!")
