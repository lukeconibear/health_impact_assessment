#!/usr/bin/env python3
import os
import re
import time
import sys
import glob
import joblib
import xarray as xr
import numpy as np
import dask.bag as db
import geopandas as gpd
import pandas as pd
from dask_jobqueue import SGECluster
from dask.distributed import Client
from numba import njit, typeof, typed, types, jit

# -----------------
# --- load data ---
# -----------------
data_path = '/nobackup/earlacoa/health/data/'
results_path = '/nobackup/earlacoa/health/example/general_global/'

res = '0.25'

with xr.open_dataset(data_path + 'gpw_v4_population_count_rev11_2020_0.25deg_crop.nc') as ds:
    pop = ds['pop']

dict_ages = xr.open_dataset(f'{data_path}GBD2019_population_2019_0.25deg.nc')

file_bm_list = []
for disease in ['lc', 'ihd', 'diab', 'str', 'ncd', 'lri', 'copd']:
    file_bm_list.extend(glob.glob(data_path + 'GBD2019_baseline_mortality*' + disease + '*' + res + 'deg.nc'))


dict_bm = {}
for file_bm_each in file_bm_list:
    with xr.open_dataset(file_bm_each) as ds:
        dict_bm_each = dict(zip([key for key in list(ds.keys())], [ds[key].values for key in list(ds.keys())]))
        dict_bm.update(dict_bm_each)


file_gemm_1 = np.load(data_path + 'GEMM_healthfunction_part1.npz')
file_gemm_2 = np.load(data_path + 'GEMM_healthfunction_part2.npz')
dict_gemm   = dict(zip([key for key in file_gemm_1], [file_gemm_1[key].astype('float32') for key in file_gemm_1]))
dict_gemm_2 = dict(zip([key for key in file_gemm_2], [file_gemm_2[key].astype('float32') for key in file_gemm_2]))
dict_gemm.update(dict_gemm_2)

# -----------------
# --- functions ---
# -----------------
def shapefile_hia(hia, measure, clips, hia_path, lat, lon, regions):
    df = pd.DataFrame({'name': regions})
    hia_list = [key for key, value in hia.items() if measure in key and "total" in key and not "yl" in key]
    hia_list.insert(0, "pop")
    if (measure == "ncdlri") or (measure == "5cod"):
        hia_list.insert(1, "pm25_popweighted")
    elif measure == "6cod":
        hia_list.insert(1, "apm25_popweighted")
    elif measure == "copd":
        hia_list.insert(1, "o3_popweighted")
    # loop through variables and regions
    for variable in hia_list:
        df[variable] = pd.Series(np.nan)
        for region in regions:
            da = xr.DataArray(hia[variable], coords=[lat, lon], dims=["lat", "lon"])
            clip = clips[region]
            da_clip = da.where(clip==0, other=np.nan) # didn't convert the values in this version to be consistent
            if variable == "pop":
                df.loc[df.name == region, variable] = np.nansum(da_clip.values)
            elif "popweighted" in variable:
                df.loc[df.name == region, variable] = (
                    np.nansum(da_clip.values) / df.loc[df.name == region, "pop"].values[0]
                )
            elif "rate" not in variable:
                df.loc[df.name == region, variable] = np.nansum(da_clip.values)
            else:
                df.loc[df.name == region, variable] = np.nanmean(da_clip.values)
    return df


def calc_hia_gemm_5cod(pm25, pop_z_2015, dict_ages, dict_bm, dict_gemm):
    """ health impact assessment using the GEMM for 5-COD """
    """ inputs are exposure to annual-mean PM2.5 on a global grid at 0.25 degrees """
    """ estimated for all ages individually, with outcomes separately """
    """ risks include China cohort """
    """ call example: calc_hia_gemm_5cod(pm25_ctl, pop_z_2015, dict_ages, dict_bm, dict_gemm) """
    # inputs
    causes = ['lri', 'lc', 'copd', 'str', 'ihd']
    ages = ['25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64',
            '65_69', '70_74', '75_79', '80up']
    outcomes = ['mort', 'yll', 'yld']
    metrics = ['mean', 'upper', 'lower']
    lcc = 2.4 # no cap at 84 ugm-3
    # health impact assessment
    hia_5cod = {}
    hia_5cod.update({'pop' : pop_z_2015})
    hia_5cod.update({'pm25_popweighted' : pop_z_2015 * pm25})
    for cause in causes:
        for outcome in outcomes:
            for metric in metrics:
                for age in ages:
                    if metric == 'mean':
                        theta = dict_gemm['gemm_health_' + cause + '_theta_' + age]
                    elif metric == 'lower':
                        theta = dict_gemm['gemm_health_' + cause + '_theta_' + age] - dict_gemm['gemm_health_' + cause + '_theta_error_' + age]
                    elif metric == 'upper':
                        theta = dict_gemm['gemm_health_' + cause + '_theta_' + age] + dict_gemm['gemm_health_' + cause + '_theta_error_' + age]
                    # mort, yll, yld - age
                    hia_5cod.update({ outcome + '_' + cause + '_' + metric + '_' + age :
                                     pop_z_2015 * dict_ages['age_fraction_' + metric + '_' + age + '_both'].values
                                     * dict_bm['i_' + outcome + '_' + cause + '_both_' + metric + '_' + age]
                                     * (1 - 1 / (np.exp(np.log(1 + (pm25 - lcc).clip(min=0)
                                                               / dict_gemm['gemm_health_' + cause + '_alpha_' + age])
                                                        / (1 + np.exp((dict_gemm['gemm_health_' + cause + '_mu_' + age]
                                                                       - (pm25 - lcc).clip(min=0))
                                                                      / dict_gemm['gemm_health_' + cause + '_pi_' + age]))
                                                        * theta))) })
                # mort - total
                hia_5cod.update({ outcome + '_' + cause + '_' + metric + '_total' :
                                 sum([value for key, value in hia_5cod.items()
                                      if outcome + '_' + cause + '_' + metric in key]) })
        # dalys - age
        for metric in metrics:
            for age in ages:
                hia_5cod.update({ 'dalys_' + cause + '_' + metric + '_' + age :
                                 hia_5cod['yll_' + cause + '_' + metric + '_' + age]
                                 + hia_5cod['yld_' + cause + '_' + metric + '_' + age] })
            # dalys - total
            hia_5cod.update({ 'dalys_' + cause + '_' + metric + '_total' :
                             sum([value for key, value in hia_5cod.items()
                                  if 'dalys_' + cause + '_' + metric in key]) })
    for outcome in ['mort', 'yll', 'yld', 'dalys']:
        for metric in metrics:
            # 5cod - total
            hia_5cod.update({ outcome + '_5cod_' + metric + '_total' :
                             sum([value for key, value in hia_5cod.items()
                                  if outcome in key and metric in key
                                  and 'total' in key and not '5cod' in key]) })
            # 5cod rates - total
            hia_5cod.update({ outcome + '_rate_5cod_' + metric + '_total' :
                             hia_5cod[outcome + '_5cod_' + metric + '_total']
                             * ( 100000 / pop_z_2015) })
    return hia_5cod


def outcome_per_age_ncdlri(pop, age_grid, age, bm_ncd, bm_lri, outcome, metric, pm25_clipped, alpha, mu, pi, theta, theta_error):
    if metric == 'mean':
        theta = theta
    elif metric == 'lower':
        theta = theta - theta_error
    elif metric == 'upper':
        theta = theta + theta_error
    return (pop * age_grid * (bm_ncd + bm_lri) * (1 - 1 / (np.exp(np.log(1 + pm25_clipped / alpha) / (1 + np.exp((mu - pm25_clipped) / pi)) * theta))))


def outcome_total(hia_ncdlri, outcome, metric):
    return sum(
        [
            value
            for key, value in hia_ncdlri.items()
            if f"{outcome}_ncdlri_{metric}" in key
        ]
    )


def dalys_age(hia_ncdlri, metric, age):
    return (
        hia_ncdlri[f"yll_ncdlri_{metric}_{age}"]
        + hia_ncdlri[f"yld_ncdlri_{metric}_{age}"]
    )


def dalys_total(hia_ncdlri, metric):
    return sum(
        [value for key, value in hia_ncdlri.items() if f"dalys_ncdlri_{metric}" in key]
    )


def rates_total(hia_ncdlri, outcome, metric, pop):
    return hia_ncdlri[f"{outcome}_ncdlri_{metric}_total"] * (100000 / pop)


def calc_hia_gemm_ncdlri(pm25, pop, dict_ages, dict_bm, dict_gemm):
    """ health impact assessment using the GEMM for NCD+LRI """
    """ inputs are exposure to annual-mean PM2.5 on a global grid at 0.25 degrees """
    """ estimated for all ages individually, with outcomes separately """
    """ risks include China cohort """
    """ call example: calc_hia_gemm_ncdlri(pm25_ctl, pop, dict_ages, dict_bm, dict_gemm) """
    # inputs
    ages = [
        "25_29",
        "30_34",
        "35_39",
        "40_44",
        "45_49",
        "50_54",
        "55_59",
        "60_64",
        "65_69",
        "70_74",
        "75_79",
        "80up",
    ]
    outcomes = ["mort", "yll", "yld"]
    metrics = ["mean", "upper", "lower"]
    lcc = 2.4  # no cap at 84 ugm-3
    pm25_clipped = (pm25 - lcc).clip(min=0)
    # health impact assessment
    hia_ncdlri = {}
    hia_ncdlri.update({"pop": pop})
    hia_ncdlri.update({"pm25_popweighted": pop * pm25})
    for outcome in outcomes:
        for metric in metrics:
            for age in ages:
                # outcome_per_age
                hia_ncdlri.update(
                    {
                        f"{outcome}_ncdlri_{metric}_{age}": outcome_per_age_ncdlri(
                            pop,
                            dict_ages['age_fraction_' + metric + '_' + age + '_both'].values,
                            age,
                            dict_bm['i_' + outcome + '_ncd_both_' + metric + '_' + age],
                            dict_bm['i_' + outcome + '_lri_both_' + metric + '_' + age],
                            outcome,
                            metric,
                            pm25_clipped,
                            dict_gemm[f"gemm_health_nonacc_alpha_{age}"],
                            dict_gemm[f"gemm_health_nonacc_mu_{age}"],
                            dict_gemm[f"gemm_health_nonacc_pi_{age}"],
                            dict_gemm[f"gemm_health_nonacc_theta_{age}"],
                            dict_gemm[f"gemm_health_nonacc_theta_error_{age}"],
                        )
                    }
                )
            # outcome_total
            hia_ncdlri.update(
                    {f"{outcome}_ncdlri_{metric}_total": outcome_total(hia_ncdlri, outcome, metric)}
            )
    for metric in metrics:
        for age in ages:
            # dalys_age
            hia_ncdlri.update(
                    {f"dalys_ncdlri_{metric}_{age}": dalys_age(hia_ncdlri, metric, age)}
            )
        # dalys_total
        hia_ncdlri.update(
            {f"dalys_ncdlri_{metric}_total": dalys_total(hia_ncdlri, metric)}
        )
    for outcome in ["mort", "yll", "yld", "dalys"]:
        for metric in metrics:
            # rates_total
            hia_ncdlri.update(
                    {f"{outcome}_rate_ncdlri_{metric}_total": rates_total(hia_ncdlri, outcome, metric, pop)}
            )
    return hia_ncdlri

# -----------------
# --- calculate ---
# -----------------
sims = ['control_2020']
for sim in sims:
    with xr.open_dataset(f'{results_path}PM2_5_DRY_{sim}_popgrid_0.25deg.nc') as ds:
        pm25 = ds['PM2_5_DRY'].values
        lon = ds.lon
        lat = ds.lat

    xx, yy = np.meshgrid(lon.values, lat.values)

    hia_ncdlri = calc_hia_gemm_ncdlri(pm25, pop, dict_ages, dict_bm, dict_gemm)
    hia_5cod   = calc_hia_gemm_5cod(pm25, pop, dict_ages, dict_bm, dict_gemm)

    hia_ncdlri_list = []
    hia_5cod_list = []

    for name, array in hia_ncdlri.items():
        ds = xr.DataArray(array, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}).to_dataset(name=name)
        hia_ncdlri_list.append(ds)


    for name, array in hia_5cod.items():
        ds = xr.DataArray(array, dims=('lat', 'lon'), coords={'lat': lat, 'lon': lon}).to_dataset(name=name)
        hia_5cod_list.append(ds)


    ds_ncdlri = xr.merge(hia_ncdlri_list)
    ds_5cod = xr.merge(hia_5cod_list)

    ds_ncdlri.to_netcdf(results_path + 'hia_ncdlri_' + sim + '.nc')
    ds_5cod.to_netcdf(results_path + 'hia_5cod_' + sim + '.nc')

    gdf = gpd.read_file(data_path + 'gadm28_adm0.shp')
    country_list = list(gdf.ID_0.values)

    df_country_hia_ncdlri = shapefile_hia(hia_ncdlri, 'ncdlri', 'country', data_path + 'gadm28_adm0.shp', results_path + '', lat, lon, regions=country_list)
    df_country_hia_5cod   = shapefile_hia(hia_5cod, '5cod', 'country', data_path + 'gadm28_adm0.shp', results_path + '', lat, lon, regions=country_list)

    df_country_hia_ncdlri.to_csv(results_path + 'df_country_hia_ncdlri_' + sim + '.csv')
    df_country_hia_5cod.to_csv(results_path + 'df_country_hia_5cod_' + sim + '.csv')

