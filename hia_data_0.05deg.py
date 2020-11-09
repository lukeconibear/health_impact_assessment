#!/usr/bin/env python3
exec(open('/nobackup/earlacoa/python/modules_python3.py').read())

res = '0.05'

# import data
data_path = '/nobackup/earlacoa/health/data/'
import_npz(data_path + 'population-count-' + res + 'deg.npz', globals())

file_age = np.load(data_path + 'GBD2017_population_age_fraction_global_2015_array_' + res + 'deg.npz')
dict_ages = dict(zip([key for key in file_age], [file_age[key].astype('float32') for key in file_age]))

file_gemm_1 = np.load(data_path + 'GEMM_healthfunction_part1.npz')
file_gemm_2 = np.load(data_path + 'GEMM_healthfunction_part2.npz')
dict_gemm   = dict(zip([key for key in file_gemm_1], [file_gemm_1[key].astype('float32') for key in file_gemm_1]))
dict_gemm_2 = dict(zip([key for key in file_gemm_2], [file_gemm_2[key].astype('float32') for key in file_gemm_2]))
dict_gemm.update(dict_gemm_2)

file_bm_list = []
for disease in ['ncd', 'lri', 'copd']:
    file_bm_list.extend(glob.glob(data_path + 'GBD2017_baseline_mortality*' + disease + '*' + res + 'deg.npz'))
dict_bm = {}
for file_bm_each in file_bm_list:
    file_bm = np.load(file_bm_each)
    dict_bm_each = dict(zip([key for key in file_bm], [file_bm[key].astype('float32') for key in file_bm]))
    dict_bm.update(dict_bm_each)

file_o3 = np.load(data_path + 'GBD2017_O3_attributablefraction.npz')
dict_o3 = dict(zip([key for key in file_o3], [file_o3[key].astype('float32') for key in file_o3]))

# gbd2017 pm2.5 ier function
file_pm25_ier = np.load(data_path + 'GBD2017_PM2.5_IER.npz')
dict_pm25_ier = dict(zip([key for key in file_pm25_ier], [file_pm25_ier[key].astype('float32') for key in file_pm25_ier]))

# solid fuel use
file_sfu  = np.load(data_path + 'shupler2018_hap2015_census2011sfu_india-state_global-grid_array.npz')
sfu = file_sfu['cf_hap_sfu_grid'].astype('float32')

