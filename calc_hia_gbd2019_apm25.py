#!/usr/bin/env python3
import numpy as np
import sys
import glob
sys.path.append('/nobackup/earlacoa/python/')
from import_npz import import_npz
from find_nearest import find_nearest

def calc_rr_gbd2019_apm25(apm25, dict_apm25):
    """
    convert PM2.5 to rr using the GBD2019 IER function for PM2.5
    call example: calc_rr_gbd2019_apm25(apm25, dict_apm25)
    """
    causes = ['lri', 'lc', 'copd', 'diab', 'str', 'ihd']
    ages = ['25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85_89', '90_94', '95up']
    uncertainties = ['mean', 'upper', 'lower']
    dict_rr = {} 
    for cause in ['lri', 'lc', 'copd', 'diab']:
        for uncertainty in uncertainties:
            dict_rr.update({
                f'{cause}_{uncertainty}':
                np.array([[dict_apm25['rr_gbd2019_' + cause + '_' + uncertainty][find_nearest(dict_apm25['rr_gbd2019_conc'], apm25[lat][lon])]
                for lon in range(apm25.shape[1])] for lat in range(apm25.shape[0])])
            })

    for cause in ['str', 'ihd']: # also split by age
        for uncertainty in uncertainties:
            for age in ages:
                dict_rr.update({
                    f'{cause}_{uncertainty}_{age}':
                    np.array([[dict_apm25['rr_gbd2019_' + cause + '_' + uncertainty + '_' + age][find_nearest(dict_apm25['rr_gbd2019_conc'], apm25[lat][lon])]
                    for lon in range(apm25.shape[1])] for lat in range(apm25.shape[0])])
                })

    return dict_rr


def calc_hia_gbd2019_apm25(apm25, dict_rr_apm25, pop_z_2015, dict_ages, dict_bm, dict_apm25):
    """
    health impact assessment using the GBD2019 IER function for PM2.5
    inputs are exposure to annual-mean ambient and household PM2.5 on a global grid at 0.25 degrees
    estimated for all ages individually
    call example: calc_hia_gbd2019_apm25(apm25, dict_rr_apm25, pop_z_2015, dict_ages, dict_bm, dict_apm25)
    """
    # inputs
    causes = ['lri', 'lc', 'copd', 'diab', 'str', 'ihd']
    ages = ['25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85_89', '90_94', '95up']
    lri_adult_ages = ['20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74', '75_79', '80_84', '85_89', '90_94', '95up']
    lri_child_ages = ['eneo', 'lneo', 'pneo', '1_4', '5_9', '10_14', '15_19']
    measures = ['mort', 'yll', 'yld']
    uncertainties = ['mean', 'upper', 'lower']
    lcc = 2.4 # no cap at 84 ugm-3
    ratio = {}
    ratio.update({'ihd': 0.141}) # for yld, rr = (ratio_ihd * rr) - ratio_ihd + 1
    ratio.update({'str': 0.553}) # for yld, rr = (ratio_str * rr) - ratio_str + 1

    # health impact assessment
    hia_apm25 = {}
    hia_apm25.update({ 'pop' : pop_z_2015 })
    hia_apm25.update({ 'apm25_popweighted' : pop_z_2015 * apm25 })
    # mort, yll, yld for age > 25
    for cause in ['lri', 'lc', 'copd', 'diab']:
        for measure in measures:
            for uncertainty in uncertainties:
                for age in ages:
                    hia_apm25.update({
                        f'{measure}_{cause}_{uncertainty}_{age}':
                        pop_z_2015
                        * dict_ages['age_fraction_' + uncertainty + '_' + age + '_both']
                        * dict_bm['i_' + measure + '_' + cause + '_both_' + uncertainty + '_' + age]
                        * (dict_rr_apm25[cause + '_' + uncertainty] - 1) / dict_rr_apm25[cause + '_' + uncertainty]
                    })

    # additional for lri 20_24 and children
    for measure in measures:
        for uncertainty in uncertainties:
            hia_apm25.update({
                measure + '_lri_' + uncertainty + '_20_24' :
                pop_z_2015
                * dict_ages['age_fraction_' + uncertainty + '_20_24_both']
                * dict_bm['i_' + measure + '_lri_both_' + uncertainty + '_20_24']
                * (dict_rr_apm25['lri_' + uncertainty] - 1) / dict_rr_apm25['lri_' + uncertainty]
            })
            for age in lri_child_ages:
                hia_apm25.update({
                    measure + '_lri_' + uncertainty + '_' + age :
                    pop_z_2015
                    * dict_ages['age_fraction_' + uncertainty + '_' + age + '_both']
                    * dict_bm['i_' + measure + '_lri_both_' + uncertainty + '_' + age]
                    * (dict_rr_apm25['lri_' + uncertainty] - 1) / dict_rr_apm25['lri_' + uncertainty]
                })

    # additional for ihd and str by age
    for cause in ['ihd', 'str']:
        for measure in ['mort', 'yll']:
            for uncertainty in uncertainties:
                for age in ages:
                    hia_apm25.update({
                        measure + '_' + cause + '_' + uncertainty + '_' + age :
                        pop_z_2015
                        * dict_ages['age_fraction_' + uncertainty + '_' + age + '_both']
                        * dict_bm['i_' + measure + '_' + cause + '_both_' + uncertainty + '_' + age]
                        * (dict_rr_apm25[cause + '_' + uncertainty + '_' + age] - 1) / dict_rr_apm25[cause + '_' + uncertainty + '_' + age]
                    })

        for measure in ['yld']: # yld with ratios for ihd and str
            for uncertainty in uncertainties:
                for age in ages:
                    hia_apm25.update({
                        measure + '_' + cause + '_' + uncertainty + '_' + age :
                        pop_z_2015
                        * dict_ages['age_fraction_' + uncertainty + '_' + age + '_both']
                        * dict_bm['i_' + measure + '_' + cause + '_both_' + uncertainty + '_' + age]
                        * ((( (ratio[cause] * dict_rr_apm25[cause + '_' + uncertainty + '_' + age]) - ratio[cause] + 1 ) - 1) / ( (ratio[cause] * dict_rr_apm25[cause + '_' + uncertainty + '_' + age]) - ratio[cause] + 1 ))
                    })

    # mort, yll, yld - totals
    for cause in causes:
        for measure in measures:
            for uncertainty in uncertainties:
                hia_apm25.update({
                    measure + '_' + cause + '_' + uncertainty + '_total' :
                    sum([value for key, value in hia_apm25.items() if measure + '_' + cause + '_' + uncertainty in key])
                })

    # dalys - total
    for cause in causes:
        for measure in measures:
            for uncertainty in uncertainties:
                hia_apm25.update({
                    'dalys_' + cause + '_' + uncertainty + '_total':
                    hia_apm25['yll_' + cause + '_' + uncertainty + '_total'] + hia_apm25['yld_' + cause + '_' + uncertainty + '_total']
                })

    # rates
    for measure in ['mort', 'yll', 'yld', 'dalys']:
        for uncertainty in uncertainties:
            # 6cod - total
            hia_apm25.update({
                measure + '_6cod_' + uncertainty + '_total' :
                sum([value for key, value in hia_apm25.items()
                    if measure in key and uncertainty in key and '6cod' not in key and 'total' in key])
            })
            # 6cod rates - total
            hia_apm25.update({
                measure + '_rate_6cod_' + uncertainty + '_total' :
                hia_apm25[measure + '_6cod_' + uncertainty + '_total'] * ( 100000 / pop_z_2015)
            })

    return hia_apm25

