#!/usr/bin/env python3
import numpy as np
import xarray as xr
import salem
from cutshapefile import transform_from_latlon, rasterize
import geopandas as gpd

# load shapefile (single multipolygon) and extract shapes
shapefile_china = gpd.read_file('/nobackup/earlacoa/emissions/shapefiles/CHN_adm0.shp')
shapes_china = [(shape, index) for index, shape in enumerate(shapefile_china.geometry)]

# load wrf dataset
ds = salem.xr.open_dataset('/path/wrf.nc')

# apply shapefile to geometry, default: inside shapefile == 0, outside shapefile == np.nan
ds['china'] = rasterize(shapes_china, ds.coords, longitude='lon', latitude='lat') 

# change to more intuitive labelling of 1 for inside shapefile and np.nan for outside shapefile
# if condition preserve (outside china, as inside defaults to 0), otherwise (1, to mark in china)
ds['china'] = ds.china.where(cond=ds.china!=0, other=1) 

# example: scale data inside shapefile by 0.5
# if condition (not in china) preserve, otherwise (in china, and scale)
ds = ds.where(cond=ds.china!=1, other=ds*0.5) 

