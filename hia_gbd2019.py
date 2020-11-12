#!/usr/bin/env python3
# libraries
exec(open('/nobackup/earlacoa/python/modules_python3.py').read())

# data
data_path = '/nobackup/earlacoa/health/data/'
results_path = '/nobackup/earlacoa/health/GBD2019/'

exec(open('/nobackup/earlacoa/python/hia_data_GBD2019_0.25deg.py').read())

with xr.open_dataset(f'{data_path}GBD2017_PM2.5_global_0.25deg_crop.nc') as ds:
    apm25 = ds['pm25'].values
    lon = ds['lon'].values
    lat = ds['lat'].values


with xr.open_dataset(f'{data_path}GBD2017_O3_global_0.25deg_crop.nc') as ds:
    o3 = ds['o3'].values


xx, yy = np.meshgrid(lon, lat)

# health impact assessment
sim = 'control'

dict_rr_apm25 = calc_rr_gbd2019_apm25(apm25, dict_apm25)

hia_apm25 = calc_hia_gbd2019_apm25(apm25, dict_rr_apm25, pop_z_2015, dict_ages, dict_bm, dict_apm25)
hia_o3 = calc_hia_gbd2019_o3(o3, pop_z_2015, dict_ages, dict_bm, dict_o3)

ds_apm25 = xr.Dataset()
ds_o3 = xr.Dataset()

for key in hia_o3.keys():
    ds_o3[key] = xr.DataArray(hia_o3[key], dims=('lat', 'lon'), coords={'lat': ds['lat'], 'lon': ds['lon']})

for key in hia_apm25.keys():
    ds_apm25[key] = xr.DataArray(hia_apm25[key], dims=('lat', 'lon'), coords={'lat': ds['lat'], 'lon': ds['lon']})


ds_apm25.to_netcdf(results_path + sim + '_gbd2019_hia_apm25.nc')
ds_o3.to_netcdf(results_path + sim + '_gbd2019_hia_o3.nc')

# cut per shapefile
# country
country_list = [49, 105, 242, 244] # China, India, UK, US
df_country_hia_apm25 = shapefile_hia(hia_apm25, '6cod', 'country', data_path + 'gadm28_adm0.shp', results_path + '', lat, lon, region_list=country_list)
df_country_hia_o3 = shapefile_hia(hia_o3, 'copd', 'country', data_path + 'gadm28_adm0.shp', results_path + '', lat, lon, region_list=country_list)
df_country_hia_apm25.to_csv(results_path + sim + '_df_country_hia_apm25.csv')
df_country_hia_o3.to_csv(results_path + sim + '_df_country_hia_o3.csv')
# state
#df_state_hia_apm25 = shapefile_hia(hia_apm25, '6cod', 'state', data_path + 'gadm36_IND_1.shp', results_path + '', lat, lon)
#df_state_hia_o3 = shapefile_hia(hia_o3, 'copd', 'state', data_path + 'gadm36_IND_1.shp', results_path + '', lat, lon)
#df_state_hia_apm25.to_csv(results_path + sim + '_df_state_hia_apm25.csv')
#df_state_hia_o3.to_csv(results_path + sim + '_df_state_hia_o3.csv')
