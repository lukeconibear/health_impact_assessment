#!/usr/bin/env python3
import xarray as xr
from import_npz import import_npz
from find_nearest import find_nearest
from calc_hia_gbd2017_pm25 import calc_hia_gbd2017_pm25, calc_rr_gbd2017_pm25
from calc_hia_gbd2017_o3 import calc_hia_gbd2017_o3
from shapefile_hia import shapefile_hia

# --- load data ---

# population count and age
import_npz('path/population-count-0.25deg.npz', globals())
file_age = np.load(data_path + 'GBD2017_population-age-0.25deg.npz')
dict_ages = dict(zip([key for key in file_age], [file_age[key].astype('float32') for key in file_age]))

# baseline mortality
file_bm_list = []
for disease in ['lc', 'ihd', 'diab', 'str', 'ncd', 'lri', 'copd']:
    file_bm_list.extend(glob.glob(data_path + 'GBD2017_baseline_mortality*' + disease + '*0.25deg.npz'))
dict_bm = {}
for file_bm_each in file_bm_list:
    file_bm = np.load(file_bm_each)
    dict_bm_each = dict(zip([key for key in file_bm], [file_bm[key].astype('float32') for key in file_bm]))
    dict_bm.update(dict_bm_each)

file_o3 = np.load(data_path + 'GBD2017_O3_attributablefraction.npz')
dict_o3 = dict(zip([key for key in file_o3], [file_o3[key].astype('float32') for key in file_o3]))

# exposure-outcome associations (IER)
file_pm25_ier = np.load(data_path + 'GBD2017_PM2.5_IER.npz')
dict_pm25_ier = dict(zip([key for key in file_pm25_ier], [file_pm25_ier[key].astype('float32') for key in file_pm25_ier]))

# exposures
with xr.open_dataset('path/ambient_pm25_o3.nc') as ds:
    ambient_pm25 = ds['PM2_5_DRY'].values
    ambient_o3 = ds['o3'].values
    lon = ds['longitude'].values
    lat = ds['latitude'].values

xx, yy = np.meshgrid(lon, lat)

with xr.open_dataset('path/household_pm25.nc') as ds:
    household_pm25_female = ds['household_pm25_female'].values
    household_pm25_male = ds['household_pm25_male'].values
    household_pm25_child = ds['household_pm25_child'].values
    solid_fuel_use = ds['solid_fuel_use'].values

exposures = {
    'ambient_pm25': ambient_pm25,
    'household_pm25': household_pm25_female,
    'household_pm25': household_pm25_male,
    'household_pm25': household_pm25_child
}

# --- health impact assessment ---

# relative risks for pm2.5
dict_rr_ambient_pm25 = calc_rr_gbd2017_pm25(
    exposures['ambient_pm25'], 
    np.zeros((np.shape(exposures['ambient_pm25'])[0], np.shape(exposures['ambient_pm25'])[1])), 
    dict_pm25_ier
)
dict_rr_total_pm25_female = calc_rr_gbd2017_pm25(
    exposures['ambient_pm25'], 
    exposures['household_pm25_female'], 
    dict_pm25_ier
)
dict_rr_total_pm25_male = calc_rr_gbd2017_pm25(
    exposures['ambient_pm25'], 
    exposures['household_pm25_male'], 
    dict_pm25_ier
)
dict_rr_total_pm25_child = calc_rr_gbd2017_pm25(
    exposures['ambient_pm25'], 
    exposures['household_pm25_child'], 
    dict_pm25_ier
)

# health impact assessment
hia_pm25 = calc_hia_gbd2017_pm25(
    exposures['ambient_pm25'], 
    exposures['household_pm25_female'], 
    exposures['household_pm25_male'], 
    exposures['household_pm25_child'],                                  
    dict_rr_ambient_pm25, 
    dict_rr_total_pm25_female, 
    dict_rr_total_pm25_male, 
    dict_rr_total_pm25_child, 
    population_count, 
    dict_ages, 
    dict_bm, 
    dict_pm25_ier, 
    solid_fuel_use
)
hia_o3 = calc_hia_gbd2017_o3(
    exposures['ambient_pm25'],
    population_count,
    dict_ages, 
    dict_bm, 
    dict_o3
)
np.savez_compressed('path/hia_gbd2017.npz', hia_pm25=hia_pm25, hia_o3=hia_o3)

# cut per shapefile
# country
country_list = [105]
df_country_hia_pm25 = shapefile_hia(
    hia_pm25, 
    '6cod', 
    'country', 
    'path/gadm28_adm0.shp',
    'results_path',
    lat,
    lon,
    region_list=country_list
)
df_country_hia_o3 = shapefile_hia(
    hia_o3,
    'copd',
    'country',
    'path/gadm28_adm0.shp',
    'path/',
    lat,
    lon,
    region_list=country_list
)
df_country_hia_pm25.to_csv('path/df_country_hia_pm25.csv')
df_country_hia_o3.to_csv('path/df_country_hia_o3.csv')
# state
df_state_hia_pm25 = shapefile_hia(
    hia_pm25, 
    '6cod', 
    'state', 
    'path/IND_adm1.shp', 
    'path', 
    lat, 
    lon
)
df_state_hia_o3 = shapefile_hia(
    hia_o3, 
    'copd', 
    'state', 
    'path/IND_adm1.shp', 
    'path', 
    lat, 
    lon
)
df_state_hia_pm25.to_csv('path/df_state_hia_pm25.csv')
df_state_hia_o3.to_csv('path/df_state_hia_o3.csv')

