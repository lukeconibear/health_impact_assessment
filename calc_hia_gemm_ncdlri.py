#!/usr/bin/env python3
import numpy as np
import glob
from import_npz import import_npz
from find_nearest import find_nearest

def outcome_per_age(pop_z_2015, dict_ages, age, dict_bm, outcome, metric, pm25_clipped, dict_gemm):
    return pop_z_2015 * dict_ages['cf_age_fraction_' + age + '_grid'] \
           * (dict_bm['i_' + outcome + '_ncd_both_' + metric + '_' + age] \
              + dict_bm['i_' + outcome + '_lri_both_' + metric + '_' + age]) \
           * (1 - 1 / (np.exp(np.log(1 + pm25_clipped \
                                     / dict_gemm['gemm_health_nonacc_alpha_' + age]) \
                              / (1 + np.exp((dict_gemm['gemm_health_nonacc_mu_' + age] \
                                             - pm25_clipped) \
                                            / dict_gemm['gemm_health_nonacc_pi_' + age])) \
                              * dict_gemm['gemm_health_nonacc_theta_' + age])))

def outcome_total(hia_ncdlri, outcome, metric):
    return sum([value for key, value in hia_ncdlri.items() if outcome + '_ncdlri_' + metric in key])

def dalys_age(hia_ncdlri, metric, age):
    return hia_ncdlri['yll_ncdlri_' + metric + '_' + age] + hia_ncdlri['yld_ncdlri_' + metric + '_' + age]

def dalys_total(hia_ncdlri, metric):
    return sum([value for key, value in hia_ncdlri.items() if 'dalys_ncdlri_' + metric in key])

def rates_total(hia_ncdlri, outcome, metric, pop_z_2015):
    return hia_ncdlri[outcome + '_ncdlri_' + metric + '_total'] * ( 100000 / pop_z_2015)

def calc_hia_gemm_ncdlri(pm25, pop_z_2015, dict_ages, dict_bm, dict_gemm):
    """ health impact assessment using the GEMM for NCD+LRI """
    """ inputs are exposure to annual-mean PM2.5 on a global grid at 0.25 degrees """
    """ estimated for all ages individually, with outcomes separately """
    """ risks include China cohort """
    """ call example: calc_hia_gemm_ncdlri(pm25_ctl, pop_z_2015, dict_ages, dict_bm, dict_gemm) """
    # inputs
    ages = ['25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59',
            '60_64', '65_69', '70_74', '75_79', '80up']
    outcomes = ['mort', 'yll', 'yld']
    metrics = ['mean', 'upper', 'lower']
    lcc = 2.4 # no cap at 84 ugm-3
    pm25_clipped = (pm25 - lcc).clip(min=0)
    # health impact assessment
    hia_ncdlri = {}
    hia_ncdlri.update({'pop' : pop_z_2015})
    hia_ncdlri.update({'pm25_popweighted' : pop_z_2015 * pm25})
    for outcome in outcomes:
        for metric in metrics:
            for age in ages:
                # outcome_per_age
                hia_ncdlri.update({ outcome + '_ncdlri_' + metric + '_' + age : outcome_per_age(pop_z_2015, dict_ages, age, dict_bm, outcome, metric, pm25_clipped, dict_gemm) })

            # outcome_total
            hia_ncdlri.update({ outcome + '_ncdlri_' + metric + '_total' : outcome_total(hia_ncdlri, outcome, metric) })

    for metric in metrics:
        for age in ages:
            # dalys_age
            hia_ncdlri.update({ 'dalys_ncdlri_' + metric + '_' + age : dalys_age(hia_ncdlri, metric, age) })

        # dalys_total
        hia_ncdlri.update({ 'dalys_ncdlri_' + metric + '_total' : dalys_total(hia_ncdlri, metric) })

    for outcome in ['mort', 'yll', 'yld', 'dalys']:
        for metric in metrics:
            # rates_total
            hia_ncdlri.update({ outcome + '_rate_ncdlri_' + metric + '_total' : rates_total(hia_ncdlri, outcome, metric, pop_z_2015) })

    return hia_ncdlri

