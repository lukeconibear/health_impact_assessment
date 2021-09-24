#!/usr/bin/env python3
import numpy as np
import glob
from import_npz import import_npz
from find_nearest import find_nearest

def calc_hia_gemm_5cod(pm25, pop_z_2015, dict_ages, dict_bm, dict_gemm):
    """ health impact assessment using the GEMM for 5-COD """
    """ inputs are exposure to annual-mean PM2.5 on a global grid at 0.25 degrees """
    """ estimated for all ages individually, with outcomes separately """
    """ risks include China cohort """
    """ call example: calc_hia_gemm_5cod(pm25_ctl, pop_z_2015, dict_ages, dict_bm, dict_gemm) """
    # inputs
    causes = ['lri', 'lc', 'copd', 'str', 'ihd']
    ages = ['25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64',
            '65_69', '70_74', '75_79', '80up']
    outcomes = ['mort', 'yll', 'yld']
    metrics = ['mean', 'upper', 'lower']
    lcc = 2.4 # no cap at 84 ugm-3
    # health impact assessment
    hia_5cod = {}
    hia_5cod.update({'pop' : pop_z_2015})
    hia_5cod.update({'pm25_popweighted' : pop_z_2015 * pm25})
    for cause in causes:
        for outcome in outcomes:
            for metric in metrics:
                for age in ages:
                    if metric == 'mean':
                        theta = dict_gemm['gemm_health_' + cause + '_theta_' + age]
                    elif metric == 'lower':
                        theta = dict_gemm['gemm_health_' + cause + '_theta_' + age] - dict_gemm['gemm_health_' + cause + '_theta_error_' + age]
                    elif metric == 'upper':
                        theta = dict_gemm['gemm_health_' + cause + '_theta_' + age] + dict_gemm['gemm_health_' + cause + '_theta_error_' + age]
                    # mort, yll, yld - age
                    hia_5cod.update({ outcome + '_' + cause + '_' + metric + '_' + age :
                                     pop_z_2015 * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_both_' + metric + '_' + age]
                                     * (1 - 1 / (np.exp(np.log(1 + (pm25 - lcc).clip(min=0)
                                                               / dict_gemm['gemm_health_' + cause + '_alpha_' + age])
                                                        / (1 + np.exp((dict_gemm['gemm_health_' + cause + '_mu_' + age]
                                                                       - (pm25 - lcc).clip(min=0))
                                                                      / dict_gemm['gemm_health_' + cause + '_pi_' + age]))
                                                        * theta))) })

                # mort - total
                hia_5cod.update({ outcome + '_' + cause + '_' + metric + '_total' :
                                 sum([value for key, value in hia_5cod.items()
                                      if outcome + '_' + cause + '_' + metric in key]) })

        # dalys - age
        for metric in metrics:
            for age in ages:
                hia_5cod.update({ 'dalys_' + cause + '_' + metric + '_' + age :
                                 hia_5cod['yll_' + cause + '_' + metric + '_' + age]
                                 + hia_5cod['yld_' + cause + '_' + metric + '_' + age] })
            # dalys - total
            hia_5cod.update({ 'dalys_' + cause + '_' + metric + '_total' :
                             sum([value for key, value in hia_5cod.items()
                                  if 'dalys_' + cause + '_' + metric in key]) })

    for outcome in ['mort', 'yll', 'yld', 'dalys']:
        for metric in metrics:
            # 5cod - total
            hia_5cod.update({ outcome + '_5cod_' + metric + '_total' :
                             sum([value for key, value in hia_5cod.items()
                                  if outcome in key and metric in key
                                  and 'total' in key and not '5cod' in key]) })
            # 5cod rates - total
            hia_5cod.update({ outcome + '_rate_5cod_' + metric + '_total' :
                             hia_5cod[outcome + '_5cod_' + metric + '_total']
                             * ( 100000 / pop_z_2015) })

    return hia_5cod
