#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd

data_path = '/path/data/'
results_path = 'path/results/'

with xr.open_dataset(results_path + 'global_0.25deg_2015-annual-mean_PM2_5_DRY.nc') as ds:
    lon = ds.lon.values
    lat = ds.lat.values

with np.load(results_path + 'hia_ncdlri.npz', allow_pickle=True) as ds:
    hia_ncdlri = ds['hia_ncdlri'][()]

    region_list = [49, 102, 132, 225]
    df_country_hia_ncdlri = shapefile_hia(
        hia_ncdlri, 
        'ncdlri', 
        'country', 
        data_path + 'gadm28_adm0.shp', 
        results_path + '', 
        lat, 
        lon, 
        region_list=region_list
    )
    df_country_hia_ncdlri.to_csv(results_path + 'df_country_hia_ncdlri.csv')

    df_province_hia_ncdlri = shapefile_hia(
        hia_ncdlri, 
        'ncdlri', 
        'province', 
        data_path + 'gadm36_CHN_1.shp', 
        results_path + '', 
        lat, 
        lon
)
    df_province_hia_ncdlri.to_csv(results_path + 'df_province_hia_ncdlri.csv')

    region_list = ['CHN.6.2_1', 'CHN.6.3_1', 'CHN.6.4_1', 'CHN.6.6_1', 'CHN.6.7_1', 'CHN.6.15_1', 'CHN.6.19_1', 'CHN.6.20_1', 'CHN.6.21_1']
    df_prefecture_hia_ncdlri = shapefile_hia(
        hia_ncdlri,
        'ncdlri', 
        'prefecture', 
        data_path + 'gadm36_CHN_2.shp', 
        results_path + '', 
        lat, 
        lon, 
        region_list=region_list
    )
    df_prefecture_hia_ncdlri.to_csv(results_path + 'df_prefecture_hia_ncdlri.csv')

