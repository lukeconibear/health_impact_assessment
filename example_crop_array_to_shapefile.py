#!/usr/bin/env python3
import numpy as np
import xarray as xr
import salem
from cutshapefile import transform_from_latlon, rasterize
import geopandas as gpd

# load shapefile (single multipolygon) and extract shapes
shapefile = gpd.read_file('/nobackup/earlacoa/emissions/shapefiles/CHN_adm0.shp')
shapes = [(shape, index) for index, shape in enumerate(shapefile.geometry)]

# load wrf dataset
ds = salem.xr.open_dataset('/path/wrf.nc')

# apply shapefile to geometry, default: inside shapefile == 0, outside shapefile == np.nan
ds['shapefile'] = rasterize(shapes, ds.coords, longitude='lon', latitude='lat') 

# change to more intuitive labelling of 1 for inside shapefile and np.nan for outside shapefile
# if condition preserve (outside shapefile, as inside defaults to 0), otherwise (1, to mark in shapefile)
ds['shapefile'] = ds.shapefile.where(cond=ds.shapefile!=0, other=1) 

# example: scale data inside shapefile by 0.5
# if condition (not in shapefile) preserve, otherwise (in shapefile, and scale)
ds = ds.where(cond=ds.shapefile!=1, other=ds*0.5) 
