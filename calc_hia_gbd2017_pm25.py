#!/usr/bin/env python3
import numpy as np
import glob
from import_npz import import_npz
from find_nearest import find_nearest

def calc_rr_gbd2017_pm25(apm25, hpm25, dict_pm25_ier):
    """ convert PM2.5 to rr using the GBD2017 IER function for PM2.5 """
    """ call example: calc_hia_gbd2017_pm25(apm25, hpm25, pop_z_2015, dict_ages, dict_bm, dict_pm25_ier, sfu) """
    causes = ['lri', 'lc', 'copd', 'diab', 'str', 'ihd']
    ages = ['25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74',
            '75_79', '80_84', '85_89', '90_94', '95up']
    metrics = ['mean', 'upper', 'lower']
    # relative risk
    # non-sfu, apm25 only (all)
    # sfu, apm25 and hpm25 (female, male, child)
    dict_rr = {} 
    for cause in ['lri', 'lc', 'copd', 'diab']:
        for metric in metrics:
            dict_rr.update({ cause + '_' + metric :
                             np.array([[dict_pm25_ier['rr_gbd2017_' + cause + '_' + metric][find_nearest(dict_pm25_ier['rr_gbd2017_conc'], apm25[lat][lon] + hpm25[lat][lon])]
                             for lon in range(apm25.shape[1])] for lat in range(apm25.shape[0])]) })

    for cause in ['str', 'ihd']: # also split by age
        for metric in metrics:
            for age in ages:
                dict_rr.update({ cause + '_' + metric + '_' + age :
                                 np.array([[dict_pm25_ier['rr_gbd2017_' + cause + '_' + metric + '_' + age][find_nearest(dict_pm25_ier['rr_gbd2017_conc'], apm25[lat][lon] + hpm25[lat][lon])]
                                 for lon in range(apm25.shape[1])] for lat in range(apm25.shape[0])]) })

    return dict_rr


def calc_hia_gbd2017_pm25(apm25, hpm25_female, hpm25_male, hpm25_child, dict_rr_apm25, dict_rr_tpm25_female, dict_rr_tpm25_male, dict_rr_tpm25_child, pop_z_2015, dict_ages, dict_bm, dict_pm25_ier, sfu):
    """ health impact assessment using the GBD2017 IER function for PM2.5 """
    """ inputs are exposure to annual-mean ambient and household PM2.5 on a global grid at 0.25 degrees """
    """ estimated for all ages individually """
    """ call example: calc_hia_gbd2017_pm25(apm25, hpm25_female, hpm25_male, hpm25_child, dict_rr_apm25, dict_rr_tpm25_female, dict_rr_tpm25_male, dict_rr_tpm25_child, pop_z_2015, dict_ages, dict_bm, dict_pm25_ier, sfu) """
    # inputs
    causes = ['lri', 'lc', 'copd', 'diab', 'str', 'ihd']
    ages = ['25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74',
            '75_79', '80_84', '85_89', '90_94', '95up']
    lri_adult_ages = ['20_24', '25_29', '30_34', '35_39', '40_44', '45_49', '50_54', '55_59', '60_64', '65_69', '70_74',
                      '75_79', '80_84', '85_89', '90_94', '95up']
    lri_child_ages = ['eneo', 'lneo', 'pneo', '1_4', '5_9', '10_14', '15_19']
    outcomes = ['mort', 'yll', 'yld']
    metrics = ['mean', 'upper', 'lower']
    lcc = 2.4 # no cap at 84 ugm-3
    ratio = {}
    ratio.update({ 'ihd':0.141 }) # for yld, rr = (ratio_ihd * rr) - ratio_ihd + 1
    ratio.update({ 'str':0.553 }) # for yld, rr = (ratio_str * rr) - ratio_str + 1
    # health impact assessment
    hia_pm25 = {}
    hia_pm25.update({ 'pop' : pop_z_2015 })
    hia_pm25.update({ 'apm25_popweighted' : pop_z_2015 * apm25 })
    hia_pm25.update({ 'hpm25_female_popweighted' : pop_z_2015 * hpm25_female })
    hia_pm25.update({ 'hpm25_male_popweighted' : pop_z_2015 * hpm25_male })
    hia_pm25.update({ 'hpm25_child_popweighted' : pop_z_2015 * hpm25_child })
    # no sfu apm25, sfu apm25 female, sfu hpm25 female, sfu apm25 male, sfu hpm25 male
    for cause in ['lri', 'lc', 'copd', 'diab']:
        for outcome in outcomes:
            for metric in metrics:
                for age in ages:
                    # no sfu - apm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_nosfu_apm25_' + cause + '_' + metric + '_' + age :
                                     (1 - sfu) # non-solid fuel using household only
                                     * pop_z_2015
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_both_' + metric + '_' + age]
                                     * (dict_rr_apm25[cause + '_' + metric] - 1) / dict_rr_apm25[cause + '_' + metric] })
                    # sfu (female) - apm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_sfu_apm25_female_' + cause + '_' + metric + '_' + age :
                                     sfu # sfu using households
                                     * (apm25 / (apm25 + hpm25_female)) # apm25 fraction of paf
                                     * (pop_z_2015 / 2) # half for female
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_female_' + metric + '_' + age]
                                     * (dict_rr_tpm25_female[cause + '_' + metric] - 1) / dict_rr_tpm25_female[cause + '_' + metric] })
                    # sfu (female) - hpm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_sfu_hpm25_female_' + cause + '_' + metric + '_' + age :
                                     sfu
                                     * (hpm25_female / (apm25 + hpm25_female))
                                     * (pop_z_2015 / 2) # half for female
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_female_' + metric + '_' + age]
                                     * (dict_rr_tpm25_female[cause + '_' + metric] - 1) / dict_rr_tpm25_female[cause + '_' + metric] })
                    # sfu (male) - apm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_sfu_apm25_male_' + cause + '_' + metric + '_' + age :
                                     sfu
                                     * (apm25 / (apm25 + hpm25_male))
                                     * (pop_z_2015 / 2) # half for male
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_male_' + metric + '_' + age]
                                     * (dict_rr_tpm25_male[cause + '_' + metric] - 1) / dict_rr_tpm25_male[cause + '_' + metric] })
                    # sfu (male) - hpm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_sfu_hpm25_male_' + cause + '_' + metric + '_' + age :
                                     sfu
                                     * (hpm25_male / (apm25 + hpm25_male))
                                     * (pop_z_2015 / 2) # half for male
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_male_' + metric + '_' + age]
                                     * (dict_rr_tpm25_male[cause + '_' + metric] - 1) / dict_rr_tpm25_male[cause + '_' + metric] })

    # additional for lri 20_24 and children
    # no sfu apm25 adult, sfu apm25 adult female, sfu hpm25 adult female, sfu apm25 adult male, sfu hpm25 adult male
    # no sfu apm25 child, sfu apm25 child, sfu hpm25 child
    for outcome in outcomes:
        for metric in metrics:
            # no sfu - apm25 - mort, yll, yld - age = 20_24
            hia_pm25.update({ outcome + '_nosfu_apm25_lri_' + metric + '_20_24' :
                             (1 - sfu)
                             * pop_z_2015
                             * dict_ages['cf_age_fraction_20_24_grid']
                             * dict_bm['i_' + outcome + '_lri_both_' + metric + '_20_24']
                             * (dict_rr_apm25['lri_' + metric] - 1) / dict_rr_apm25['lri_' + metric] })
            # sfu (female) - apm25 - mort, yll, yld - age = 20_24
            hia_pm25.update({ outcome + '_sfu_apm25_female_lri_' + metric + '_20_24' :
                             sfu
                             * (apm25 / (apm25 + hpm25_female))
                             * (pop_z_2015 / 2) # half for female
                             * dict_ages['cf_age_fraction_20_24_grid']
                             * dict_bm['i_' + outcome + '_lri_female_' + metric + '_20_24']
                             * (dict_rr_tpm25_female['lri_' + metric] - 1) / dict_rr_tpm25_female['lri_' + metric] })
            # sfu (female) - hpm25 - mort, yll, yld - age = 20_24
            hia_pm25.update({ outcome + '_sfu_hpm25_female_lri_' + metric + '_20_24' :
                             sfu
                             * (hpm25_female / (apm25 + hpm25_female))
                             * (pop_z_2015 / 2) # half for female
                             * dict_ages['cf_age_fraction_20_24_grid']
                             * dict_bm['i_' + outcome + '_lri_female_' + metric + '_20_24']
                             * (dict_rr_tpm25_female['lri_' + metric] - 1) / dict_rr_tpm25_female['lri_' + metric] })
            # sfu (male) - apm25 - mort, yll, yld - age = 20_24
            hia_pm25.update({ outcome + '_sfu_apm25_male_lri_' + metric + '_20_24' :
                             sfu
                             * (apm25 / (apm25 + hpm25_male))
                             * (pop_z_2015 / 2) # half for male
                             * dict_ages['cf_age_fraction_20_24_grid']
                             * dict_bm['i_' + outcome + '_lri_male_' + metric + '_20_24']
                             * (dict_rr_tpm25_male['lri_' + metric] - 1) / dict_rr_tpm25_male['lri_' + metric] })
            # sfu (male) - hpm25 - mort, yll, yld - age = 20_24
            hia_pm25.update({ outcome + '_sfu_hpm25_male_lri_' + metric + '_20_24' :
                             sfu
                             * (hpm25_male / (apm25 + hpm25_male))
                             * (pop_z_2015 / 2) # half for male
                             * dict_ages['cf_age_fraction_20_24_grid']
                             * dict_bm['i_' + outcome + '_lri_male_' + metric + '_20_24']
                             * (dict_rr_tpm25_male['lri_' + metric] - 1) / dict_rr_tpm25_male['lri_' + metric] })
            for age in lri_child_ages:
                # no sfu - apm25 - mort, yll, yld - age < 20
                hia_pm25.update({ outcome + '_nosfu_apm25_lri_' + metric + '_' + age :
                                 (1 - sfu)
                                 * pop_z_2015
                                 * dict_ages['cf_age_fraction_' + age + '_grid']
                                 * dict_bm['i_' + outcome + '_lri_both_' + metric + '_' + age]
                                 * (dict_rr_apm25['lri_' + metric] - 1) / dict_rr_apm25['lri_' + metric] })
                # sfu (child) - apm25 - mort, yll, yld - age < 20
                hia_pm25.update({ outcome + '_sfu_apm25_child_lri_' + metric + '_' + age :
                                 sfu
                                 * (apm25 / (apm25 + hpm25_child))
                                 * pop_z_2015
                                 * dict_ages['cf_age_fraction_' + age + '_grid']
                                 * dict_bm['i_' + outcome + '_lri_both_' + metric + '_' + age]
                                 * (dict_rr_tpm25_child['lri_' + metric] - 1) / dict_rr_tpm25_child['lri_' + metric] })
                # sfu (child) - hpm25 - mort, yll, yld - age < 20
                hia_pm25.update({ outcome + '_sfu_hpm25_child_lri_' + metric + '_' + age :
                                 sfu
                                 * (hpm25_child / (apm25 + hpm25_child))
                                 * pop_z_2015
                                 * dict_ages['cf_age_fraction_' + age + '_grid']
                                 * dict_bm['i_' + outcome + '_lri_both_' + metric + '_' + age]
                                 * (dict_rr_tpm25_child['lri_' + metric] - 1) / dict_rr_tpm25_child['lri_' + metric] })

    # no sfu apm25, sfu apm25 female, sfu hpm25 female, sfu apm25 male, sfu hpm25 male
    for cause in ['ihd', 'str']:
        for outcome in ['mort', 'yll']:
            for metric in metrics:
                for age in ages:
                    # no sfu - apm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_nosfu_apm25_' + cause + '_' + metric + '_' + age :
                                     (1 - sfu)
                                     * pop_z_2015
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_both_' + metric + '_' + age]
                                     * (dict_rr_apm25[cause + '_' + metric + '_' + age] - 1) / dict_rr_apm25[cause + '_' + metric + '_' + age] })
                    # sfu (female) - apm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_sfu_apm25_female_' + cause + '_' + metric + '_' + age :
                                     sfu
                                     * (apm25 / (apm25 + hpm25_female))
                                     * (pop_z_2015 / 2) # half for female
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_female_' + metric + '_' + age]
                                     * (dict_rr_tpm25_female[cause + '_' + metric + '_' + age] - 1) / dict_rr_tpm25_female[cause + '_' + metric + '_' + age] })
                    # sfu (female) - hpm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_sfu_hpm25_female_' + cause + '_' + metric + '_' + age :
                                     sfu
                                     * (hpm25_female / (apm25 + hpm25_female))
                                     * (pop_z_2015 / 2) # half for female
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_female_' + metric + '_' + age]
                                     * (dict_rr_tpm25_female[cause + '_' + metric + '_' + age] - 1) / dict_rr_tpm25_female[cause + '_' + metric + '_' + age] })
                    # sfu (male) - apm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_sfu_apm25_male_' + cause + '_' + metric + '_' + age :
                                     sfu
                                     * (apm25 / (apm25 + hpm25_male))
                                     * (pop_z_2015 / 2) # half for male
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_male_' + metric + '_' + age]
                                     * (dict_rr_tpm25_male[cause + '_' + metric + '_' + age] - 1) / dict_rr_tpm25_male[cause + '_' + metric + '_' + age] })
                    # sfu (male) - hpm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_sfu_hpm25_male_' + cause + '_' + metric + '_' + age :
                                     sfu
                                     * (hpm25_male / (apm25 + hpm25_male))
                                     * (pop_z_2015 / 2) # half for male
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_male_' + metric + '_' + age]
                                     * (dict_rr_tpm25_male[cause + '_' + metric + '_' + age] - 1) / dict_rr_tpm25_male[cause + '_' + metric + '_' + age] })

    # yld with ratios for ihd and str
    for cause in ['ihd', 'str']:
        for outcome in ['yld']:
            for metric in metrics:
                for age in ages:
                    # no sfu - apm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_nosfu_apm25_' + cause + '_' + metric + '_' + age :
                                     (1 - sfu)
                                     * pop_z_2015
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_both_' + metric + '_' + age]
                                     * ((( (ratio[cause] * dict_rr_apm25[cause + '_' + metric + '_' + age]) - ratio[cause] + 1 ) - 1) / ( (ratio[cause] * dict_rr_apm25[cause + '_' + metric + '_' + age]) - ratio[cause] + 1 )) })
                    # sfu (female) - apm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_sfu_apm25_female_' + cause + '_' + metric + '_' + age :
                                     sfu
                                     * (apm25 / (apm25 + hpm25_female))
                                     * (pop_z_2015 / 2) # half for female
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_female_' + metric + '_' + age]
                                     * ((( (ratio[cause] * dict_rr_tpm25_female[cause + '_' + metric + '_' + age]) - ratio[cause] + 1 ) - 1) / ( (ratio[cause] * dict_rr_tpm25_female[cause + '_' + metric + '_' + age]) - ratio[cause] + 1 )) })
                    # sfu (female) - hpm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_sfu_hpm25_female_' + cause + '_' + metric + '_' + age :
                                     sfu
                                     * (hpm25_female / (apm25 + hpm25_female))
                                     * (pop_z_2015 / 2) # half for female
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_female_' + metric + '_' + age]
                                     * ((( (ratio[cause] * dict_rr_tpm25_female[cause + '_' + metric + '_' + age]) - ratio[cause] + 1 ) - 1) / ( (ratio[cause] * dict_rr_tpm25_female[cause + '_' + metric + '_' + age]) - ratio[cause] + 1 )) })
                    # sfu (male) - apm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_sfu_apm25_male_' + cause + '_' + metric + '_' + age :
                                     sfu
                                     * (apm25 / (apm25 + hpm25_male))
                                     * (pop_z_2015 / 2) # half for male
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_male_' + metric + '_' + age]
                                     * ((( (ratio[cause] * dict_rr_tpm25_male[cause + '_' + metric + '_' + age]) - ratio[cause] + 1 ) - 1) / ( (ratio[cause] * dict_rr_tpm25_male[cause + '_' + metric + '_' + age]) - ratio[cause] + 1 )) })
                    # sfu (male) - hpm25 - mort, yll, yld - age > 25
                    hia_pm25.update({ outcome + '_sfu_hpm25_male_' + cause + '_' + metric + '_' + age :
                                     sfu
                                     * (hpm25_male / (apm25 + hpm25_male))
                                     * (pop_z_2015 / 2) # half for male
                                     * dict_ages['cf_age_fraction_' + age + '_grid']
                                     * dict_bm['i_' + outcome + '_' + cause + '_male_' + metric + '_' + age]
                                     * ((( (ratio[cause] * dict_rr_tpm25_male[cause + '_' + metric + '_' + age]) - ratio[cause] + 1 ) - 1) / ( (ratio[cause] * dict_rr_tpm25_male[cause + '_' + metric + '_' + age]) - ratio[cause] + 1 )) })

    # cata
    for metric in metrics:
        for age in ages:
            # sfu (female) - hpm25 - yld - age > 25
            hia_pm25.update({ 'yld_sfu_hpm25_female_cata_' + metric + '_' + age :
                             sfu
                             * (pop_z_2015 / 2) # half for female
                             * dict_ages['cf_age_fraction_' + age + '_grid']
                             * dict_bm['i_yld_cata_female_' + metric + '_' + age]
                             * ((1.0247 * hpm25_female / 10 ) - 1 ) / ( 1.0247 * hpm25_female / 10 ) })

    # mort, yll, yld - totals
    for cause in causes:
        for outcome in outcomes:
            for metric in metrics:
                # apm25
                # no sfu
                hia_pm25.update({ outcome + '_nosfu_apm25_' + cause + '_' + metric + '_total' :
                                 sum([value for key, value in hia_pm25.items()
                                      if outcome + '_nosfu_apm25_' + cause + '_' + metric in key]) })
                # sfu - female
                hia_pm25.update({ outcome + '_sfu_apm25_female_' + cause + '_' + metric + '_total' :
                                 sum([value for key, value in hia_pm25.items()
                                      if outcome + '_sfu_apm25_female_' + cause + '_' + metric in key]) })
                # sfu - male
                hia_pm25.update({ outcome + '_sfu_apm25_male_' + cause + '_' + metric + '_total' :
                                 sum([value for key, value in hia_pm25.items()
                                      if outcome + '_sfu_apm25_male_' + cause + '_' + metric in key]) })
                # sfu - child
                hia_pm25.update({ outcome + '_sfu_apm25_child_' + cause + '_' + metric + '_total' :
                                 sum([value for key, value in hia_pm25.items()
                                      if outcome + '_sfu_apm25_child_' + cause + '_' + metric in key]) })
                # sfu - total
                hia_pm25.update({ outcome + '_sfu_apm25_' + cause + '_' + metric + '_total' :
                                 hia_pm25[outcome + '_sfu_apm25_female_' + cause + '_' + metric + '_total']
                                 + hia_pm25[outcome + '_sfu_apm25_male_' + cause + '_' + metric + '_total']
                                 + hia_pm25[outcome + '_sfu_apm25_child_' + cause + '_' + metric + '_total'] })
                # apm25 - total
                hia_pm25.update({ outcome + '_apm25_' + cause + '_' + metric + '_total' :
                                 hia_pm25[outcome + '_sfu_apm25_' + cause + '_' + metric + '_total']
                                 + hia_pm25[outcome + '_nosfu_apm25_' + cause + '_' + metric + '_total'] })

                # sfu - hpm25 - female
                hia_pm25.update({ outcome + '_sfu_hpm25_female_' + cause + '_' + metric + '_total' :
                                 sum([value for key, value in hia_pm25.items()
                                      if outcome + '_sfu_hpm25_female_' + cause + '_' + metric in key]) })
                # sfu - hpm25 - male
                hia_pm25.update({ outcome + '_sfu_hpm25_male_' + cause + '_' + metric + '_total' :
                                 sum([value for key, value in hia_pm25.items()
                                      if outcome + '_sfu_hpm25_male_' + cause + '_' + metric in key]) })
                # sfu - hpm25 - child
                hia_pm25.update({ outcome + '_sfu_hpm25_child_' + cause + '_' + metric + '_total' :
                                 sum([value for key, value in hia_pm25.items()
                                      if outcome + '_sfu_hpm25_child_' + cause + '_' + metric in key]) })
                # sfu - hpm25
                hia_pm25.update({ outcome + '_hpm25_' + cause + '_' + metric + '_total' :
                                 hia_pm25[outcome + '_sfu_hpm25_female_' + cause + '_' + metric + '_total']
                                 + hia_pm25[outcome + '_sfu_hpm25_male_' + cause + '_' + metric + '_total']
                                 + hia_pm25[outcome + '_sfu_hpm25_child_' + cause + '_' + metric + '_total'] })
                # tpm25
                hia_pm25.update({ outcome + '_tpm25_' + cause + '_' + metric + '_total' :
                                hia_pm25[outcome + '_apm25_' + cause + '_' + metric + '_total']
                                 + hia_pm25[outcome + '_hpm25_' + cause + '_' + metric + '_total'] })

    # dalys - total
    for split in ['apm25', 'hpm25', 'tpm25']:
        for cause in causes:
            for outcome in outcomes:
                for metric in metrics:
                    hia_pm25.update({ 'dalys_' + split + '_' + cause + '_' + metric + '_total':
                                     hia_pm25['yll_' + split + '_' + cause + '_' + metric + '_total']
                                     + hia_pm25['yld_' + split + '_' + cause + '_' + metric + '_total'] })

    for split in ['apm25', 'hpm25', 'tpm25']:
        for outcome in ['mort', 'yll', 'yld', 'dalys']:
            for metric in metrics:
                # 6cod - total
                hia_pm25.update({ outcome + '_' + split + '_6cod_' + metric + '_total' :
                                 sum([value for key, value in hia_pm25.items()
                                      if outcome in key and metric in key
                                      and split in key and 'total' in key
                                      and not '6cod' in key
                                      and not 'female' in key
                                      and not 'male' in key
                                      and not 'child' in key
                                      and not 'nosfu' in key
                                      and not 'sfu' in key]) })
                # 6cod rates - total
                hia_pm25.update({ outcome + '_' + split + '_rate_6cod_' + metric + '_total' :
                                 hia_pm25[outcome + '_' + split + '_6cod_' + metric + '_total']
                                 * ( 100000 / pop_z_2015) })

    return hia_pm25
