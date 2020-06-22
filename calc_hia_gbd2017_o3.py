#!/usr/bin/env python3
import numpy as np
import glob
from import_npz import import_npz
from find_nearest import find_nearest

def calc_hia_gbd2017_o3(o3, pop_z_2015, dict_ages, dict_bm, dict_o3):
    """ health impact assessment using the GBD2017 function for O3 """
    """ inputs are exposure to annual-mean, daily maximum, 8-hour, O3 concentrations (ADM8h) on a global grid at 0.25 degrees """
    """ estimated for all ages individually """
    """ call example: calc_hia_gbd2017_o3(o3_ctl, pop_z_2015, dict_ages, dict_bm, dict_o3) """
    # inputs
    ages = ['25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64',
            '65_69', '70_74', '75_79', '80up']
    outcomes = ['mort', 'yll', 'yld']
    metrics = ['mean', 'upper', 'lower']
    # health impact assessment
    hia_o3 = {}
    hia_o3.update({'pop' : pop_z_2015})
    hia_o3.update({'o3_popweighted' : pop_z_2015 * o3})

    # attributable fraction
    dict_af = {}
    for metric in metrics:
        dict_af.update({ metric : np.array([[dict_o3['af_o3_copd_' + metric][find_nearest(dict_o3['o3_conc'], o3[lat][lon])] for lon in range(o3.shape[1])] for lat in range(o3.shape[0])]) })

    for outcome in outcomes:
        for metric in metrics:
            for age in ages:
                # mort, yll, yld - age
                hia_o3.update({ outcome + '_copd_' + metric + '_' + age :
                                 pop_z_2015 * dict_ages['cf_age_fraction_' + age + '_grid']
                                 * dict_bm['i_' + outcome + '_copd_both_' + metric + '_' + age]
                                 * dict_af[metric] })

            # mort, yll, yld - total
            hia_o3.update({ outcome + '_copd_' + metric + '_total' :
                             sum([value for key, value in hia_o3.items()
                                  if outcome + '_copd_' + metric in key]) })

    # dalys - age
    for metric in metrics:
        for age in ages:
            hia_o3.update({ 'dalys_copd_' + metric + '_' + age :
                             hia_o3['yll_copd_' + metric + '_' + age]
                             + hia_o3['yld_copd_' + metric + '_' + age] })
        # dalys - total
        hia_o3.update({ 'dalys_copd_' + metric + '_total' :
                         sum([value for key, value in hia_o3.items()
                              if 'dalys_copd_' + metric in key]) })

    # rates - total
    for outcome in ['mort', 'yll', 'yld', 'dalys']:
        for metric in metrics:
            hia_o3.update({ outcome + '_rate_copd_' + metric + '_total' :
                             hia_o3[outcome + '_copd_' + metric + '_total']
                             * ( 100000 / pop_z_2015) })

    return hia_o3
