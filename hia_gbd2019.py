#!/usr/bin/env python3
# libraries
exec(open('/nobackup/earlacoa/python/modules_python3.py').read())

# data
data_path = '/nobackup/earlacoa/health/data/'
results_path = '/nobackup/earlacoa/health/GBD2019/'

exec(open('/nobackup/earlacoa/python/hia_data_GBD2019_0.25deg.py').read())

with xr.open_dataset(f'{data_path}GBD2017_PM2.5_global_0.25deg_crop.nc') as ds:
    pm25 = ds['pm25'].values
    lon = ds['lon'].values
    lat = ds['lat'].values


with xr.open_dataset(f'{data_path}GBD2017_O3_global_0.25deg_crop.nc') as ds:
    o3 = ds['o3'].values


xx, yy = np.meshgrid(lon, lat)

# health impact assessment
sim = 'control'

dict_rr_pm25 = calc_rr_gbd2019_pm25(pm25, dict_pm25)

hia_pm25 = calc_hia_gbd2019_pm25(pm25, dict_rr_pm25, pop_z_2015, dict_ages, dict_bm, dict_pm25)
hia_o3 = calc_hia_gbd2019_o3(o3, pop_z_2015, dict_ages, dict_bm, dict_o3)

np.savez_compressed(results_path + sim + '_hia_gbd2019.npz', hia_pm25=hia_pm25, hia_o3=hia_o3)

# cut per shapefile
# country
country_list = [105]
df_country_hia_pm25 = shapefile_hia(hia_pm25, '6cod', 'country', data_path + 'gadm28_adm0.shp', results_path + '', lat, lon, region_list=country_list)
df_country_hia_o3 = shapefile_hia(hia_o3, 'copd', 'country', data_path + 'gadm28_adm0.shp', results_path + '', lat, lon, region_list=country_list)
df_country_hia_pm25.to_csv(results_path + sim + '_df_country_hia_pm25.csv')
df_country_hia_o3.to_csv(results_path + sim + '_df_country_hia_o3.csv')
# state
df_state_hia_pm25 = shapefile_hia(hia_pm25, '6cod', 'state', data_path + 'gadm36_IND_1.shp', results_path + '', lat, lon)
df_state_hia_o3 = shapefile_hia(hia_o3, 'copd', 'state', data_path + 'gadm36_IND_1.shp', results_path + '', lat, lon)
df_state_hia_pm25.to_csv(results_path + sim + '_df_state_hia_pm25.csv')
df_state_hia_o3.to_csv(results_path + sim + '_df_state_hia_o3.csv')

# to open npz dict of arrays
#hia_gemm = np.load(results_path + 'hia_gemm.npz', allow_pickle=True)
#hia_ncdlri_ctl  = hia_gemm['hia_ncdlri_ctl'][()]
#hia_ncdlri_nobb = hia_gemm['hia_ncdlri_nobb'][()]
#hia_5cod_ctl  = hia_gemm['hia_5cod_ctl'][()]
#hia_5cod_nobb = hia_gemm['hia_5cod_nobb'][()]
