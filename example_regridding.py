#!/usr/bin/env python3
import xarray as xr
import xesmf as xe
import salem

with salem.xr.open_dataset('/path/wrf.nc') as ds:
    wrf = ds['variable']
    
global_grid = xr.Dataset(
    {'lat': (['lat'], np.arange(-60, 85, 0.0416667)), 
     'lon': (['lon'], np.arange(-180, 180, 0.0416667)),}
)

regridder = xe.Regridder(
    wrf, 
    global_grid, 
    'bilinear', 
    reuse_weights=True
)

wrf_regridded = regridder(wrf)

regridder.clean_weight_file()
