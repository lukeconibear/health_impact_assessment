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

# ------------------------------------

sensitivity_pop2020 = False
sensitivity_age2020 = False
sensitivity_bm2020 = False
sensitivity_popagebm2020 = False

if sensitivity_pop2020:
    sensitivity_folder = 'health_impact_assessments_sens-pop2020'
elif sensitivity_age2020:
    sensitivity_folder = 'health_impact_assessments_sens-age2020'
elif sensitivity_bm2020:
    sensitivity_folder = 'health_impact_assessments_sens-bm2020'
elif sensitivity_popagebm2020:
    sensitivity_folder = 'health_impact_assessments_sens-popagebm2020'
else:
    sensitivity_folder = 'health_impact_assessments'


clips = joblib.load('/nobackup/earlacoa/health/example/china_climate/clips_0.125deg.joblib')

dict_pops = {
    '2020': xr.open_dataset('/nobackup/earlacoa/health/data/projections/SSP2/Total/NetCDF/ssp2_2020.nc')['ssp2_2020'].values.astype("float32"),
    '2030': xr.open_dataset('/nobackup/earlacoa/health/data/projections/SSP2/Total/NetCDF/ssp2_2030.nc')['ssp2_2030'].values.astype("float32"),
    '2040': xr.open_dataset('/nobackup/earlacoa/health/data/projections/SSP2/Total/NetCDF/ssp2_2040.nc')['ssp2_2040'].values.astype("float32"),
    '2050': xr.open_dataset('/nobackup/earlacoa/health/data/projections/SSP2/Total/NetCDF/ssp2_2050.nc')['ssp2_2050'].values.astype("float32")}

if sensitivity_pop2020 or sensitivity_popagebm2020:
    dict_pops = {
        '2020': xr.open_dataset('/nobackup/earlacoa/health/data/projections/SSP2/Total/NetCDF/ssp2_2020.nc')['ssp2_2020'].values.astype("float32"),
        '2030': xr.open_dataset('/nobackup/earlacoa/health/data/projections/SSP2/Total/NetCDF/ssp2_2020.nc')['ssp2_2020'].values.astype("float32"),
        '2040': xr.open_dataset('/nobackup/earlacoa/health/data/projections/SSP2/Total/NetCDF/ssp2_2020.nc')['ssp2_2020'].values.astype("float32"),
        '2050': xr.open_dataset('/nobackup/earlacoa/health/data/projections/SSP2/Total/NetCDF/ssp2_2020.nc')['ssp2_2020'].values.astype("float32")}

with xr.open_dataset('/nobackup/earlacoa/health/data/projections/SSP2/Total/NetCDF/ssp2_2020.nc')['ssp2_2020'] as ds:
    lon = ds['lon']
    lat = ds['lat']


pop_xx, pop_yy = np.meshgrid(lon, lat)

dict_ages = {}
age_files = glob.glob('/nobackup/earlacoa/health/data/IFs/IFs_pop_age_china_*.nc')
for age_file in age_files:
    age = '_'.join(re.findall(r'\d+', age_file)[1:])
    if age == '80':
        age = '80up'
    year = re.findall(r'\d+', age_file)[0]
    with xr.open_dataset(age_file) as ds:
        values = ds['Total'].values.astype("float32")
    dict_ages.update({f'{year}_{age}': values})


if sensitivity_age2020 or sensitivity_popagebm2020:
    for key in sorted(list(dict_ages.keys())):
        if '2030' in key or '2040' in key or '2050' in key:
            dict_ages[key] = dict_ages[f'2020_{key[5:]}']


dict_bm = {}
bm_files = glob.glob('/nobackup/earlacoa/health/data/IFs/IFs_bm*mort_*.nc')
for bm_file in bm_files:
    age = '_'.join(re.findall(r'\d+', bm_file)[1:])
    if age == '80':
        age = '80up'
    year = re.findall(r'\d+', bm_file)[0]
    for possible_disease in ['ncd', 'lri', 'copd']:
        if re.findall(r'' + possible_disease + '', bm_file):
            disease = possible_disease
    outcome = 'mort'
    with xr.open_dataset(bm_file) as ds:
        key = [key for key in ds.keys()][0]
        values = ds[key].values.astype("float32")
    dict_bm.update({f'{year}_{age}_{outcome}_{disease}': values})


if sensitivity_bm2020 or sensitivity_popagebm2020:
    for key in sorted(list(dict_bm.keys())):
        if '2030' in key or '2040' in key or '2050' in key:
            dict_bm[key] = dict_bm[f'2020_{key[5:]}']


dict_af = {}
dict_af.update({'mean':  joblib.load('/nobackup/earlacoa/health/data/o3_dict_af_mean.joblib')})
dict_af.update({'lower': joblib.load('/nobackup/earlacoa/health/data/o3_dict_af_lower.joblib')})
dict_af.update({'upper': joblib.load('/nobackup/earlacoa/health/data/o3_dict_af_upper.joblib')})

with np.load(
    "/nobackup/earlacoa/health/data/GEMM_healthfunction_part1.npz"
) as file_gemm_1:
    dict_gemm = dict(
        zip(
            [key for key in file_gemm_1],
            [np.atleast_1d(file_gemm_1[key].astype("float32")) for key in file_gemm_1],
        )
    )

with np.load(
    "/nobackup/earlacoa/health/data/GEMM_healthfunction_part2.npz"
) as file_gemm_2:
    dict_gemm_2 = dict(
        zip(
            [key for key in file_gemm_2],
            [np.atleast_1d(file_gemm_2[key].astype("float32")) for key in file_gemm_2],
        )
    )

dict_gemm.update(dict_gemm_2)

# ------------------------------------

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


def dict_to_typed_dict(dict_normal):
    """convert to typed dict for numba"""
    if len(dict_normal[next(iter(dict_normal))].shape) == 1:
        value_shape = types.f4[:]
    elif len(dict_normal[next(iter(dict_normal))].shape) == 2:
        value_shape = types.f4[:, :]
    typed_dict = typed.Dict.empty(types.string, value_shape)
    for key, value in dict_normal.items():
        typed_dict[key] = value
    return typed_dict


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


def calc_hia_gemm_ncdlri(pm25, pop, dict_ages, dict_bm, dict_gemm, year):
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
    outcomes = ["mort"] # , "yll", "yld"
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
                            dict_ages[f"{year}_{age}"],
                            age,
                            dict_bm[f"{year}_{age}_{outcome}_ncd"],
                            dict_bm[f"{year}_{age}_{outcome}_lri"],
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

    #for metric in metrics:
    #    for age in ages:
    #        # dalys_age
    #        hia_ncdlri.update(
    #                {f"dalys_ncdlri_{metric}_{age}": dalys_age(hia_ncdlri, metric, age)}
    #        )

        # dalys_total
    #    hia_ncdlri.update(
    #        {f"dalys_ncdlri_{metric}_total": dalys_total(hia_ncdlri, metric)}
    #    )

    for outcome in ["mort"]: # , "yll", "yld", "dalys"
        for metric in metrics:
            # rates_total
            hia_ncdlri.update(
                    {f"{outcome}_rate_ncdlri_{metric}_total": rates_total(hia_ncdlri, outcome, metric, pop)}
            )

    return hia_ncdlri


def create_attribute_fraction(value, dict_af):
    return dict_af[f'{value}']


create_attribute_fraction = np.vectorize(create_attribute_fraction)


def calc_hia_gbd2017_o3(o3, pop, dict_ages, dict_bm, dict_af, year):
    """ health impact assessment using the GBD2017 function for O3 """
    """ inputs are exposure to annual-mean, daily maximum, 8-hour, O3 concentrations (ADM8h) on a global grid at 0.25 degrees """
    """ estimated for all ages individually """
    """ call example: calc_hia_gbd2017_o3(o3_ctl, pop, dict_ages, dict_bm, dict_af) """
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
    outcomes = ["mort"] # , "yll", "yld"
    metrics = ["mean", "upper", "lower"]
    # health impact assessment
    hia_o3 = {}
    hia_o3.update({"pop": pop})
    hia_o3.update({"o3_popweighted": pop * o3})

    # attributable fraction
    o3_rounded = np.nan_to_num(np.around(o3, 1)) # 1dp for the nearest af
    af = {}
    for metric in metrics:
        af.update({
            metric: create_attribute_fraction(o3_rounded, dict_af[metric])
        })

    for outcome in outcomes:
        for metric in metrics:
            for age in ages:
                # mort, yll, yld - age
                hia_o3.update(
                    {
                        f"{outcome}_copd_{metric}_{age}": pop
                        * dict_ages[f"{year}_{age}"]
                        * dict_bm[f"{year}_{age}_{outcome}_copd"]
                        * af[metric]
                    }
                )

            # mort, yll, yld - total
            hia_o3.update(
                {
                    f"{outcome}_copd_{metric}_total": sum(
                        [
                            value
                            for key, value in hia_o3.items()
                            if f"{outcome}_copd_{metric}" in key
                        ]
                    )
                }
            )

    # dalys - age
    #for metric in metrics:
    #    for age in ages:
    #        hia_o3.update(
    #            {
    #                f"dalys_copd_{metric}_{age}": hia_o3[f"yll_copd_{metric}_{age}"]
    #                + hia_o3[f"yld_copd_{metric}_{age}"]
    #            }
    #        )
        # dalys - total
    #    hia_o3.update(
    #        {
    #            f"dalys_copd_{metric}_total": sum(
    #                [
    #                    value
    #                    for key, value in hia_o3.items()
    #                    if f"dalys_copd_{metric}" in key
    #                ]
    #            )
    #        }
    #    )

    # rates - total
    for outcome in ["mort"]: #, "yll", "yld", "dalys"
        for metric in metrics:
            hia_o3.update(
                {
                    f"{outcome}_rate_copd_{metric}_total": hia_o3[f"{outcome}_copd_{metric}_total"] 
                    * (100_000 / pop)
                }
            )

    return hia_o3



def health_impact_assessment_pm25(simulation):
    with xr.open_dataset(f"/nobackup/earlacoa/health/example/china_climate/exposures/{simulation}_0.125deg.nc") as ds:
        output = simulation[-9:]
        pm25 = ds[output].values
        lon = ds.lon.values
        lat = ds.lat.values

    xx, yy = np.meshgrid(lon, lat)

    year = re.findall(r'\d+', simulation)[0]

    hia_ncdlri = calc_hia_gemm_ncdlri(pm25, dict_pops[year], dict_ages, dict_bm, dict_gemm, year)
    np.savez_compressed(
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/hia_{simulation}.npz",
        hia_ncdlri=hia_ncdlri,
    )

    countries = ['China', 'Hong Kong', 'Macao', 'Taiwan']
    provinces = ['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong', 'Guangxi', 'Guizhou', 'Hainan', 'Hebei', 'Heilongjiang', 'Henan', 'Hubei', 'Hunan', 'Jiangsu', 'Jiangxi', 'Jilin', 'Liaoning', 'Nei Mongol', 'Ningxia Hui', 'Qinghai', 'Shaanxi', 'Shandong', 'Shanghai', 'Shanxi', 'Sichuan', 'Tianjin', 'Xinjiang Uygur', 'Xizang', 'Yunnan', 'Zhejiang']
    prefectures = ['Dongguan', 'Foshan', 'Guangzhou', 'Huizhou', 'Jiangmen', 'Shenzhen', 'Zhaoqing', 'Zhongshan', 'Zhuhai']
    
    df_country_hia_ncdlri = shapefile_hia(
        hia_ncdlri,
        "ncdlri",
        clips,
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/",
        lat,
        lon,
        countries,
    )
    df_country_hia_ncdlri.to_csv(
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/df_country_hia_{simulation}.csv"
    )

    df_province_hia_ncdlri = shapefile_hia(
        hia_ncdlri,
        "ncdlri",
        clips,
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/",
        lat,
        lon,
        provinces,
    )
    df_province_hia_ncdlri.to_csv(
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/df_province_hia_{simulation}.csv"
    )

    df_prefecture_hia_ncdlri = shapefile_hia(
        hia_ncdlri,
        "ncdlri",
        clips,
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/",
        lat,
        lon,
        prefectures,
    )
    df_prefecture_hia_ncdlri.to_csv(
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/df_prefecture_hia_{simulation}.csv"
    )


def health_impact_assessment_o3(simulation):
    with xr.open_dataset(f"/nobackup/earlacoa/health/example/china_climate/exposures/{simulation}_0.125deg.nc") as ds:
        output = simulation[-9:]
        o3_6mDM8h = ds[output].values
        lon = ds.lon.values
        lat = ds.lat.values

    xx, yy = np.meshgrid(lon, lat)

    year = re.findall(r'\d+', simulation)[0]

    hia_o3 = calc_hia_gbd2017_o3(o3_6mDM8h, dict_pops[year], dict_ages, dict_bm, dict_af, year)
    np.savez_compressed(
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/hia_{simulation}.npz",
        hia_o3=hia_o3,
    )

    countries = ['China', 'Hong Kong', 'Macao', 'Taiwan']
    provinces = ['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong', 'Guangxi', 'Guizhou', 'Hainan', 'Hebei', 'Heilongjiang', 'Henan', 'Hubei', 'Hunan', 'Jiangsu', 'Jiangxi', 'Jilin', 'Liaoning', 'Nei Mongol', 'Ningxia Hui', 'Qinghai', 'Shaanxi', 'Shandong', 'Shanghai', 'Shanxi', 'Sichuan', 'Tianjin', 'Xinjiang Uygur', 'Xizang', 'Yunnan', 'Zhejiang']
    prefectures = ['Dongguan', 'Foshan', 'Guangzhou', 'Huizhou', 'Jiangmen', 'Shenzhen', 'Zhaoqing', 'Zhongshan', 'Zhuhai']
    
    df_country_hia_o3 = shapefile_hia(
        hia_o3,
        "copd",
        clips,
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/",
        lat,
        lon,
        countries,
    )
    df_country_hia_o3.to_csv(
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/df_country_hia_{simulation}.csv"
    )

    df_province_hia_o3 = shapefile_hia(
        hia_o3,
        "copd",
        clips,
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/",
        lat,
        lon,
        provinces,
    )
    df_province_hia_o3.to_csv(
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/df_province_hia_{simulation}.csv"
    )

    df_prefecture_hia_o3 = shapefile_hia(
        hia_o3,
        "copd",
        clips,
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/",
        lat,
        lon,
        prefectures,
    )
    df_prefecture_hia_o3.to_csv(
        f"/nobackup/earlacoa/health/example/china_climate/{sensitivity_folder}/df_prefecture_hia_{simulation}.csv"
    )



