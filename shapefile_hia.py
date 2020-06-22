#!/usr/bin/env python3
import numpy as np
import geopandas as gpd
import pandas as pd
import xarray as xr
from rasterio import features
from affine import Affine
import glob
from import_npz import import_npz
from find_nearest import find_nearest
from cutshapefile import transform_from_latlon, rasterize

def shapefile_hia(hia, measure, region, shapefile_file, hia_path, lat, lon, **kwargs):
    """ cut the health impact assessment arrays by shapefile """
    """ measure is ncdlri or 5cod for gemm, 6cod for ier, and copd for o3"""
    # create dataframe
    shp = gpd.read_file(shapefile_file)
    if region == 'country':
        ids = list(shp['ID_0'].values)
        names = list(shp['NAME_ENGLI'].values)
    elif (region == 'state') or (region == 'province'):
        ids = list(shp['GID_1'].values)
        names = list(shp['NAME_1'].values)
    elif (region == 'city') or (region == 'prefecture'):
        ids = list(shp['GID_2'].values)
        names = list(shp['NAME_2'].values)

    names = [name.replace(' ', '_') for name in names]
    df = pd.DataFrame(np.asarray(ids))
    df.columns = ['id']
    df['name'] = pd.Series(np.asarray(names))
    region_list = kwargs.get('region_list', None)
    if region_list != None:
        df = df.loc[df['id'].isin(region_list),:]
        ids = np.array(region_list)

    hia_list = [key for key, value in hia.items() if measure in key and 'total' in key and not 'yl' in key]
    hia_list.insert(0, 'pop')
    if (measure == 'ncdlri') or (measure == '5cod'):
        hia_list.insert(1, 'pm25_popweighted')
    elif measure == '6cod':
        hia_list.insert(1, 'apm25_popweighted')
        hia_list.insert(2, 'hpm25_female_popweighted')
        hia_list.insert(3, 'hpm25_male_popweighted')
        hia_list.insert(4, 'hpm25_child_popweighted')
    elif measure == 'copd':
        hia_list.insert(1, 'o3_popweighted')

    # loop through variables and regions
    for variable in hia_list:
        df[variable] = pd.Series(np.nan)
        for i in ids:
            # create list of tuples (shapely.geometry, id) to allow for many different polygons within a .shp file
            if region == 'country':
                shapes = [(shape, n) for n, shape in enumerate(shp[shp.ID_0 == i].geometry)]
            elif (region == 'state') or (region == 'province'):
                shapes = [(shape, n) for n, shape in enumerate(shp[shp.GID_1 == i].geometry)]
            elif (region == 'city') or (region == 'prefecture'):
                shapes = [(shape, n) for n, shape in enumerate(shp[shp.GID_2 == i].geometry)]
                
            # create dataarray for each variable
            da = xr.DataArray(hia[variable], coords=[lat, lon], dims=['lat', 'lon'])
            # create the clip for the shapefile
            clip = rasterize(shapes, da.coords, longitude='lon', latitude='lat')
            # clip the dataarray
            da_clip = da.where(clip==0, other=np.nan)
            # assign to dataframe
            if variable == 'pop':
                df.loc[df.id == i, variable] = np.nansum(da_clip.values)

            elif 'popweighted' in variable:
                df.loc[df.id == i, variable] = np.nansum(da_clip.values) / df.loc[df.id == i, 'pop'].values[0]

            elif 'rate' not in variable:
                df.loc[df.id == i, variable] = np.nansum(da_clip.values)

            else:
                df.loc[df.id == i, variable] = np.nanmean(da_clip.values)

    return df
