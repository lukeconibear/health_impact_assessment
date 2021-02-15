#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd

# --- load data ---

# population count and age
import_npz('path/population-count-0.25deg.npz', globals())

# baseline mortality
path = '/nfs/a68/earlacoa/health/GBD2017/'
df_bm = pd.read_csv(f'{path}/GBD2017_BM_2015_countries_cause_deaths-DALYS-YLL-YLD_rateper100000_LRI-LC-COPD-IHD-DIAB-STR-NCD-ALL-CATA.csv')
location = 163
baseline_mortality_annual = float(df_bm.loc[df_bm['location_id'] == location][df_bm['year'] == 2015][df_bm['age_id'] == 22][df_bm['cause_id'] == 294][df_bm['measure_id'] == 1][df_bm['sex_id'] == 3]['val']) / 100000
baseline_mortality_daily = baseline_mortality_annual / 365.25

# short-term ambient PM2.5 exposure
with xr.open_dataset(f'{path}/ambient_pm25_o3.nc') as ds:
    ambient_pm25 = ds['PM2_5_DRY'].values
    lon = ds['longitude'].values
    lat = ds['latitude'].values

xx, yy = np.meshgrid(lon, lat)

# exposure-outcome association
# Atkinson, R. W., Kang, S., Anderson, H. R., Mills, I. C., & Walton, H. A. (2014). Epidemiological time series studies of PM2.5 and daily mortality and hospital admissions: a systematic review and meta-analysis. Thorax, 69(7), 660665. https://doi.org/10.1136/thoraxjnl-2013-204492
# percentage risk = 1.04% per 10 ugm-3 from Atkinson et al., (2014), which as relative risk = 1.0104 (95CI: 0.52, 1.56)
# for comparison to the WHO 2013 HRAPIE: percentage risk = 1.23% per 10 ugm-3 from WHO 2013 HRAPIE, which as relative risk = 1.0123 (95CI: 0.45%, 2.01%) 
pm25_percentage_risk_per10_mean  = 1.04
pm25_percentage_risk_per10_lower = 0.52
pm25_percentage_risk_per10_upper = 1.56
# no lower concentration cutoff (lcc) = 0 ug/m3 (i.e. none)

# --- health impact assessment ---

# daily mean mortality, all-cause, all ages

# linear exposure-outcome association
relative_risk_linnocap_mean = 1 + ((pm25_percentage_risk_per10_mean / 100) * (ambient_pm25 / 10)
attributable_fraction_linnocap_mean = (relative_risk_linnocap_mean - 1) / relative_risk_linnocap_mean
mortality_linnocap_mean = baseline_mortality_daily * population_count * attributable_fraction_linnocap_mean

# logarithmic exposure-outcome association
relative_risk_lognocap_mean = np.exp(((pm25_percentage_risk_per10_mean / 100) * (ambient_pm25 / 10))
attributable_fraction_lognocap_mean = (relative_risk_lognocap_mean - 1) / relative_risk_lognocap_mean
mort_lognocap_mean = baseline_mortality_daily * population_count * attributable_fraction_lognocap_mean

