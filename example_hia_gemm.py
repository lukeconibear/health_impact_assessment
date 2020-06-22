#!/usr/bin/env python3
exec(open('/nobackup/earlacoa/python/modules_python3.py').read())

# --- load data ---

# population count and age
import_npz('path/population-count-0.25deg.npz', globals())

# baseline mortality
with np.load(data_path + 'GBD2017_population_age_fraction_global_2015_array_0.25deg.npz') as file_age:
    dict_ages = dict(zip([key for key in file_age], 
                         [file_age[key].astype('float32') for key in file_age]))

typed_dict_ages = dict_to_typed_dict(dict_ages)
del dict_ages

# exposure-outcome association
with np.load(data_path + 'GEMM_healthfunction_part1.npz') as file_gemm_1:
    dict_gemm   = dict(zip([key for key in file_gemm_1], 
                           [np.atleast_1d(file_gemm_1[key].astype('float32')) for key in file_gemm_1]))

with np.load(data_path + 'GEMM_healthfunction_part2.npz') as file_gemm_2:
    dict_gemm_2 = dict(zip([key for key in file_gemm_2], 
                           [np.atleast_1d(file_gemm_2[key].astype('float32')) for key in file_gemm_2]))

dict_gemm.update(dict_gemm_2)
typed_dict_gemm = dict_to_typed_dict(dict_gemm)
del dict_gemm, dict_gemm_2

# --- health impact assessment ---

outcomes = ['mort', 'yll', 'yld']
metrics = ['mean', 'upper', 'lower']
ages = ['25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80up']
diseases = ['ncd', 'lri']
lcc = 2.4 # no cap at 84 ugm-3

results_path = 'path/'
with xr.open_dataset(results_path + 'global_0.25deg_2015-annual-mean_PM2_5_DRY.nc') as ds:
    pm25 = ds['PM2_5_DRY'].values
    lon = ds.lon.values
    lat = ds.lat.values

xx, yy = np.meshgrid(lon, lat)
pm25_clipped = (pm25 - lcc).clip(min=0)

hia_ncdlri = {}
hia_ncdlri.update({'pop' : pop_z_2015})
hia_ncdlri.update({'pm25_popweighted' : pop_z_2015 * pm25})

for outcome in outcomes:
    dict_bm = load_bm_per_outcome(data_path, res, outcome, diseases)
    typed_dict_bm = dict_to_typed_dict(dict_bm)
    del dict_bm

    for metric in metrics:
        for age in ages:
            # outcome_per_age
            hia_ncdlri.update({
                outcome + '_ncdlri_' + metric + '_' + age : 
                outcome_per_age(pop_z_2015, typed_dict_ages, age, typed_dict_bm,
                                outcome, metric, pm25_clipped, typed_dict_gemm) 
            })
            
    del typed_dict_bm

for outcome in outcomes:
    for metric in metrics:
        # outcome_total
        hia_ncdlri.update({ 
            outcome + '_ncdlri_' + metric + '_total' : 
            outcome_total(hia_ncdlri, outcome, metric) 
        })

for metric in metrics:
    for age in ages:
        # dalys_age
        hia_ncdlri.update({ 
            'dalys_ncdlri_' + metric + '_' + age : 
            dalys_age(hia_ncdlri, metric, age) 
        })

    # dalys_total
    hia_ncdlri.update({ 
        'dalys_ncdlri_' + metric + '_total' : 
        dalys_total(hia_ncdlri, metric) 
    })

for outcome in ['mort', 'yll', 'yld', 'dalys']:
    for metric in metrics:
        # rates_total
        hia_ncdlri.update({ 
            outcome + '_rate_ncdlri_' + metric + '_total' : 
            rates_total(hia_ncdlri, outcome, metric, pop_z_2015) 
        })

np.savez_compressed(results_path + 'hia_ncdlri.npz', hia_ncdlri=hia_ncdlri)

