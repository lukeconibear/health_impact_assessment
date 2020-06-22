#!/usr/bin/env python3
import xarray as xr
import xesmf as xe

with xr.open_dataset('/path/file.nc') as ds:
    variable = ds['variable']
    
global_grid = xr.Dataset(
    {'lat': (['lat'], np.arange(-60, 85, 0.0416667)), 
     'lon': (['lon'], np.arange(-180, 180, 0.0416667)),}
)

regridder = xe.Regridder(
	variable, 
	global_grid, 
	'bilinear', 
	reuse_weights=True
)

variable_regridded = regridder(variable)

regridder.clean_weight_file()
