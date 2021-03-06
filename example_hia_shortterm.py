#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd

# --- load data ---
path = '/data'

# population count 
# global data as a xarray dataset following: create_population_data.ipynb
with xr.open_dataset(f'{path}/gpw_v4_population_count_rev11_2020_0.25deg_crop.nc') as ds:
    population_count = ds['pop']
    
# or could use the numpy array version instead
# import_npz('path/population-count-0.25deg.npz', globals())

# baseline mortality
df_bm = pd.read_csv(f'{path}/GBD2017_BM_2015_countries_cause_deaths-DALYS-YLL-YLD_rateper100000_LRI-LC-COPD-IHD-DIAB-STR-NCD-ALL-CATA.csv')
location = 163
baseline_mortality_annual_mean  = float(df_bm.loc[df_bm['location_id'] == location][df_bm['year'] == 2015][df_bm['age_id'] == 22][df_bm['cause_id'] == 294][df_bm['measure_id'] == 1][df_bm['sex_id'] == 3]['val']) / 100000
baseline_mortality_annual_upper = float(df_bm.loc[df_bm['location_id'] == location][df_bm['year'] == 2015][df_bm['age_id'] == 22][df_bm['cause_id'] == 294][df_bm['measure_id'] == 1][df_bm['sex_id'] == 3]['upper']) / 100000
baseline_mortality_annual_lower = float(df_bm.loc[df_bm['location_id'] == location][df_bm['year'] == 2015][df_bm['age_id'] == 22][df_bm['cause_id'] == 294][df_bm['measure_id'] == 1][df_bm['sex_id'] == 3]['lower']) / 100000
baseline_mortality_daily_mean  = baseline_mortality_annual_mean / 365.25
baseline_mortality_daily_upper = baseline_mortality_annual_upper / 365.25
baseline_mortality_daily_lower = baseline_mortality_annual_lower / 365.25

# short-term ambient PM2.5 exposure
# regridded to global grid using: example_regridding.py
with xr.open_dataset(f'{path}/ambient_pm25_o3.nc') as ds:
    # pm25
    # uses daily-mean exposures
    # no lower concentration cutoff (lcc) = 0 ug/m3 (i.e. none)
    ambient_pm25 = ds['PM2_5_DRY'].resample(time='24H').mean().values
    
    # o3
    # uses daily-maximum, 8-hourly-mean exposures
    # uses lower concentration cutoff (lcc) = 10 ppb
    ambient_o3 = ds['o3']
    ambient_o3 = ambient_o3.resample(time='8H').mean().resample(time='24H').max()
    ambient_o3 = ambient_o3.where(cond=ambient_o3 > 10, other=0.0).values
    
    lon = ds['longitude'].values
    lat = ds['latitude'].values

xx, yy = np.meshgrid(lon, lat)

# exposure-outcome association - pm25
# Liu et al., (2019) NEJM, https://doi.org/10.1056/nejmoa1817364
# percentage risk = 0.68% (95CI: 0.59, 0.77) per 10 ugm-3
# as relative risk = 1.0068 (95CI: 1.0059, 1.0077) per 10 ugm-3
pm25_percentage_risk_per10_mean  = 0.68
pm25_percentage_risk_per10_lower = 0.59
pm25_percentage_risk_per10_upper = 0.77

# exposure-outcome association - o3
# Héroux et al., (2015) IJPH, https://link.springer.com/article/10.1007/s00038-015-0690-y
# percentage risk = 0.29% (95CI: 0.14, 0.43) per 10 ppb
# as relative risk = 1.0029 (95CI: 1.0014, 1.0043) per 10 ppb
o3_percentage_risk_per10_mean  = 0.29
o3_percentage_risk_per10_lower = 0.14
o3_percentage_risk_per10_upper = 0.43

# --- health impact assessment ---
# daily mean mortality, all-cause, all ages
                                    
# pm25
# logarithmic exposure-outcome association
relative_risk_pm25_lognocap_mean = np.exp(((pm25_percentage_risk_per10_mean / 100) * (ambient_pm25 / 10))
relative_risk_pm25_lognocap_lower = np.exp(((pm25_percentage_risk_per10_lower / 100) * (ambient_pm25 / 10))
relative_risk_pm25_lognocap_upper = np.exp(((pm25_percentage_risk_per10_upper / 100) * (ambient_pm25 / 10))
attributable_fraction_pm25_lognocap_mean = (relative_risk_pm25_lognocap_mean - 1) / relative_risk_pm25_lognocap_mean
attributable_fraction_pm25_lognocap_lower = (relative_risk_pm25_lognocap_lower - 1) / relative_risk_pm25_lognocap_lower
attributable_fraction_pm25_lognocap_upper = (relative_risk_pm25_lognocap_upper - 1) / relative_risk_pm25_lognocap_upper
mortality_pm25_lognocap_mean = baseline_mortality_daily * population_count * attributable_fraction_pm25_lognocap_mean
mortality_pm25_lognocap_lower = baseline_mortality_daily * population_count * attributable_fraction_pm25_lognocap_lower
mortality_pm25_lognocap_upper = baseline_mortality_daily * population_count * attributable_fraction_pm25_lognocap_upper

# o3
# logarithmic exposure-outcome association
relative_risk_o3_lognocap_mean = np.exp(((o3_percentage_risk_per10_mean / 100) * (ambient_o3 / 10))
relative_risk_o3_lognocap_lower = np.exp(((o3_percentage_risk_per10_lower / 100) * (ambient_o3 / 10))
relative_risk_o3_lognocap_upper = np.exp(((o3_percentage_risk_per10_upper / 100) * (ambient_o3 / 10))
attributable_fraction_o3_lognocap_mean = (relative_risk_o3_lognocap_mean - 1) / relative_risk_o3_lognocap_mean
attributable_fraction_o3_lognocap_lower = (relative_risk_o3_lognocap_lower - 1) / relative_risk_o3_lognocap_lower
attributable_fraction_o3_lognocap_upper = (relative_risk_o3_lognocap_upper - 1) / relative_risk_o3_lognocap_upper
mortality_o3_lognocap_mean = baseline_mortality_daily * population_count * attributable_fraction_o3_lognocap_mean
mortality_o3_lognocap_lower = baseline_mortality_daily * population_count * attributable_fraction_o3_lognocap_lower
mortality_o3_lognocap_upper = baseline_mortality_daily * population_count * attributable_fraction_o3_lognocap_upper