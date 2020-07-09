#!/usr/bin/env python3
"""
Create daily mean surface ambient PM2.5 concentrations from raw WRFChem wrfout files
Regrid to a global regrid
"""
import glob
import numpy as np
import xarray as xr
import xesmf as xe
import salem

path = '/nfs/b0122/Users/earlacoa/test/'
wrfout_files = sorted(glob.glob(path + '/wrfout_*'))

# combine pm25 from multiple wrfout files together
with salem.open_mf_wrf_dataset(wrfout_files, chunks={'west_east':'auto', 'south_north':'auto'}) as ds:
    wrf_pm25 = ds['PM2_5_DRY']
    
wrf_pm25_surface = wrf_pm25.isel(bottom_top=0)
wrf_pm25_surface_dailymean = wrf_pm25_surface.resample(time='24H').mean()

global_grid = xr.Dataset(
    {'lat': (['lat'], np.arange(-60, 85, 0.0416667)), 
     'lon': (['lon'], np.arange(-180, 180, 0.0416667)),}
)

regridder = xe.Regridder(
    wrf_pm25_surface_dailymean, 
    global_grid, 
    'bilinear'
)

wrf_pm25_surface_dailymean_regridded = regridder(wrf_pm25_surface_dailymean)
