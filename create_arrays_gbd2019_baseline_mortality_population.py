#!/usr/bin/env python
# coding=utf-8
import shapefile
import pandas as pd
import xarray as xr
import numpy as np

res = '0.25'
year = 2019
process_baseline_mortality = True
process_population = True

# setup global shapefile
data_path = '/nobackup/earlacoa/health/data'

df_shape_array = pd.read_csv(f'{data_path}/gadm28_adm0_{res}deg_by_name_with_set_latlon_crop.csv')
df_shape_id = pd.read_csv(f'{data_path}/gadm28_adm0_{res}deg_metadata.csv')

df_shape_array["country_name"] = np.nan
for country_number in range(1,257):
    df_shape_array.loc[df_shape_array['country'] == country_number, 'country_name'] = str(np.asarray(df_shape_id.loc[df_shape_id['ID_0'] == country_number, 'NAME_ENGLI'])[0])


sf = shapefile.Reader(f'{data_path}/gadm28_adm0')
country_ids = []
country_names = []
for shape_rec in sf.shapeRecords():
    country_ids.extend([shape_rec.record[1]])
    country_names.extend([shape_rec.record[3]])

    
country_names.remove('Akrotiri and Dhekelia')
country_names.remove(u'\xc5land')
country_names.remove('Anguilla')
country_names.remove('Antarctica')
country_names.remove('Aruba')
country_names = [country.replace('Bahamas', 'The Bahamas') for country in country_names]
country_names.remove('Bonaire, Saint Eustatius and Saba')
country_names.remove('Bouvet Island')
country_names.remove('British Indian Ocean Territory')
country_names.remove('British Virgin Islands')
country_names.remove('Caspian Sea')
country_names.remove('Cayman Islands')
country_names.remove('Christmas Island')
country_names.remove('Clipperton Island')
country_names.remove('Cocos Islands')
country_names.remove('Cook Islands')
country_names = [country.replace(u"C\xf4te d'Ivoire", 'Ivory Coast') for country in country_names]
country_names = [country.replace('Republic of Congo', 'Congo') for country in country_names]
country_names.remove(u'Cura\xe7ao')
country_names.remove('Falkland Islands')
country_names.remove('Faroe Islands')
country_names.remove('French Guiana')
country_names.remove('French Polynesia')
country_names.remove('French Southern Territories')
country_names.remove('Gambia')
country_names.remove('Gibraltar')
country_names.remove('Guadeloupe')
country_names.remove('Guernsey')
country_names.remove('Heard Island and McDonald Islands')
country_names.remove('Hong Kong')
country_names.remove('Macao')
country_names.remove('Isle of Man')
country_names.remove('Jersey')
country_names.remove('Kosovo')
country_names.remove('Liechtenstein')
country_names.remove('Martinique')
country_names.remove('Mayotte')
country_names.remove('Micronesia')
country_names.remove('Monaco')
country_names.remove('Montserrat')
country_names.remove('Nauru')
country_names.remove('New Caledonia')
country_names.remove('Niue')
country_names.remove('Norfolk Island')
country_names.remove('Northern Cyprus')
country_names.remove('Palau')
country_names.remove('Palestina')
country_names.remove('Paracel Islands')
country_names.remove('Pitcairn Islands')
country_names.remove('Reunion')
country_names.remove(u'Saint-Barth\xe9lemy')
country_names.remove('Saint-Martin')
country_names.remove('Saint Helena')
country_names.remove('Saint Kitts and Nevis')
country_names.remove('Saint Pierre and Miquelon')
country_names.remove('San Marino')
country_names.remove('Sint Maarten')
country_names.remove('South Georgia and the South Sandwich Islands')
country_names.remove('Spratly islands')
country_names.remove('Svalbard and Jan Mayen')
country_names.remove('East Timor')
country_names.remove('Tokelau')
country_names.remove('Turks and Caicos Islands')
country_names.remove('Tuvalu')
country_names.remove('United States Minor Outlying Islands')
country_names.remove('Vatican City')
country_names.remove('Wallis and Futuna')
country_names.remove('Western Sahara')

# df_shape_array and df_shape_array need to have matching country names
df_shape_id.replace({"Cote d Ivoire": "Ivory Coast"}, inplace=True)
df_shape_array.replace({"Cote d Ivoire": "Ivory Coast"}, inplace=True)
df_shape_id.replace({"Russia": "Russia"}, inplace=True)
df_shape_array.replace({"Russia": "Russia"}, inplace=True)
df_shape_id.replace({"Republic of Congo": "Congo"}, inplace=True)
df_shape_array.replace({"Republic of Congo": "Congo"}, inplace=True)
df_shape_id.replace({"Akrotiri and Dhekelia": "Cyprus"}, inplace=True)
df_shape_array.replace({"Akrotiri and Dhekelia": "Cyprus"}, inplace=True)
df_shape_id.replace({"Åland": "Finland"}, inplace=True)
df_shape_array.replace({"Åland": "Finland"}, inplace=True)
df_shape_id.replace({"Anguilla": "Caribbean"}, inplace=True)
df_shape_array.replace({"Anguilla": "Caribbean"}, inplace=True)
df_shape_id.replace({"Aruba": "Caribbean"}, inplace=True)
df_shape_array.replace({"Aruba": "Caribbean"}, inplace=True)
df_shape_id.replace({"Bahamas": "The Bahamas"}, inplace=True)
df_shape_array.replace({"Bahamas": "The Bahamas"}, inplace=True)
df_shape_id.replace({"Bonaire, Saint Eustatius and Saba": "Caribbean"}, inplace=True)
df_shape_array.replace({"Bonaire, Saint Eustatius and Saba": "Caribbean"}, inplace=True)
df_shape_id.replace({"Bouvet Island": "Norway"}, inplace=True)
df_shape_array.replace({"Bouvet Island": "Norway"}, inplace=True)
df_shape_id.replace({"British Indian Ocean Territory": "India"}, inplace=True)
df_shape_array.replace({"British Indian Ocean Territory": "India"}, inplace=True)
df_shape_id.replace({"British Virgin Islands": "Caribbean"}, inplace=True)
df_shape_array.replace({"British Virgin Islands": "Caribbean"}, inplace=True)
df_shape_id.replace({"Cayman Islands": "Caribbean"}, inplace=True)
df_shape_array.replace({"Cayman Islands": "Caribbean"}, inplace=True)
df_shape_id.replace({"Christmas Island": "Indonesia"}, inplace=True)
df_shape_array.replace({"Christmas Island": "Indonesia"}, inplace=True)
df_shape_id.replace({"Clipperton Island": "Mexico"}, inplace=True)
df_shape_array.replace({"Clipperton Island": "Mexico"}, inplace=True)
df_shape_id.replace({"Cocos Islands": "Australia"}, inplace=True)
df_shape_array.replace({"Cocos Islands": "Australia"}, inplace=True)
df_shape_id.replace({"Cook Islands": "Fiji"}, inplace=True)
df_shape_array.replace({"Cook Islands": "Fiji"}, inplace=True)
df_shape_id.replace({"Curaçao": "Caribbean"}, inplace=True)
df_shape_array.replace({"Curaçao": "Caribbean"}, inplace=True)
df_shape_id.replace({"Falkland Islands": "Argentina"}, inplace=True)
df_shape_array.replace({"Falkland Islands": "Argentina"}, inplace=True)
df_shape_id.replace({"Faroe Islands": "Denmark"}, inplace=True)
df_shape_array.replace({"Faroe Islands": "Denmark"}, inplace=True)
df_shape_id.replace({"French Polynesia": "Caribbean"}, inplace=True)
df_shape_array.replace({"French Polynesia": "Caribbean"}, inplace=True)
df_shape_id.replace({"French Guiana": "Caribbean"}, inplace=True)
df_shape_array.replace({"French Guiana": "Caribbean"}, inplace=True)
df_shape_id.replace({"Gambia": "Senegal"}, inplace=True)
df_shape_array.replace({"Gambia": "Senegal"}, inplace=True)
df_shape_id.replace({"Gibraltar": "Spain"}, inplace=True)
df_shape_array.replace({"Gibraltar": "Spain"}, inplace=True)
df_shape_id.replace({"Guadeloupe": "Caribbean"}, inplace=True)
df_shape_array.replace({"Guadeloupe": "Caribbean"}, inplace=True)
df_shape_id.replace({"Guernsey": "France"}, inplace=True)
df_shape_array.replace({"Guernsey": "France"}, inplace=True)
df_shape_id.replace({"Hong Kong": "China"}, inplace=True)
df_shape_array.replace({"Hong Kong": "China"}, inplace=True)
df_shape_id.replace({"Macao": "China"}, inplace=True)
df_shape_array.replace({"Macao": "China"}, inplace=True)
df_shape_id.replace({"Isle of Man": "Ireland"}, inplace=True)
df_shape_array.replace({"Isle of Man": "Ireland"}, inplace=True)
df_shape_id.replace({"Jersey": "France"}, inplace=True)
df_shape_array.replace({"Jersey": "France"}, inplace=True)
df_shape_id.replace({"Kosovo": "Serbia"}, inplace=True)
df_shape_array.replace({"Kosovo": "Serbia"}, inplace=True)
df_shape_id.replace({"Liechtenstein": "Austria"}, inplace=True)
df_shape_array.replace({"Liechtenstein": "Austria"}, inplace=True)
df_shape_id.replace({"Martinique": "Caribbean"}, inplace=True)
df_shape_array.replace({"Martinique": "Caribbean"}, inplace=True)
df_shape_id.replace({"Mayotte": "Madagascar"}, inplace=True)
df_shape_array.replace({"Mayotte": "Madagascar"}, inplace=True)
df_shape_id.replace({"Micronesia": "Indonesia"}, inplace=True)
df_shape_array.replace({"Micronesia": "Indonesia"}, inplace=True)
df_shape_id.replace({"Monaco": "France"}, inplace=True)
df_shape_array.replace({"Monaco": "France"}, inplace=True)
df_shape_id.replace({"Montserrat": "Caribbean"}, inplace=True)
df_shape_array.replace({"Montserrat": "Caribbean"}, inplace=True)
df_shape_id.replace({"Nauru": "Indonesia"}, inplace=True)
df_shape_array.replace({"Nauru": "Indonesia"}, inplace=True)
df_shape_id.replace({"New Caledonia": "Indonesia"}, inplace=True)
df_shape_array.replace({"New Caledonia": "Indonesia"}, inplace=True)
df_shape_id.replace({"Niue": "Caribbean"}, inplace=True)
df_shape_array.replace({"Niue": "Caribbean"}, inplace=True)
df_shape_id.replace({"Norfolk Island": "Australia"}, inplace=True)
df_shape_array.replace({"Norfolk Island": "Australia"}, inplace=True)
df_shape_id.replace({"Northern Cyprus": "Cyprus"}, inplace=True)
df_shape_array.replace({"Northern Cyprus": "Cyprus"}, inplace=True)
df_shape_id.replace({"Palau": "Indonesia"}, inplace=True)
df_shape_array.replace({"Palau": "Indonesia"}, inplace=True)
df_shape_id.replace({"Palestina": "Israel"}, inplace=True)
df_shape_array.replace({"Palestina": "Israel"}, inplace=True)
df_shape_id.replace({"Reunion": "Madagascar"}, inplace=True)
df_shape_array.replace({"Reunion": "Madagascar"}, inplace=True)
df_shape_id.replace({"Saint-Barthélemy": "Caribbean"}, inplace=True)
df_shape_array.replace({"Saint-Barthélemy": "Caribbean"}, inplace=True)
df_shape_id.replace({"Saint-Martin": "Caribbean"}, inplace=True)
df_shape_array.replace({"Saint-Martin": "Caribbean"}, inplace=True)
df_shape_id.replace({"Saint Kitts and Nevis": "Caribbean"}, inplace=True)
df_shape_array.replace({"Saint Kitts and Nevis": "Caribbean"}, inplace=True)
df_shape_id.replace({"Saint Pierre and Miquelon": "Canada"}, inplace=True)
df_shape_array.replace({"Saint Pierre and Miquelon": "Canada"}, inplace=True)
df_shape_id.replace({"San Marino": "Italy"}, inplace=True)
df_shape_array.replace({"San Marino": "Italy"}, inplace=True)
df_shape_id.replace({"Sint Maarten": "Caribbean"}, inplace=True)
df_shape_array.replace({"Sint Maarten": "Caribbean"}, inplace=True)
df_shape_id.replace({"South Georgia and the South Sandwich Islands": "Argentina"}, inplace=True)
df_shape_array.replace({"South Georgia and the South Sandwich Islands": "Argentina"}, inplace=True)
df_shape_id.replace({"Svalbard and Jan Mayen": "Norway"}, inplace=True)
df_shape_array.replace({"Svalbard and Jan Mayen": "Norway"}, inplace=True)
df_shape_id.replace({"East Timor": "Indonesia"}, inplace=True)
df_shape_array.replace({"East Timor": "Indonesia"}, inplace=True)
df_shape_id.replace({"Turks and Caicos Islands": "Cuba"}, inplace=True)
df_shape_array.replace({"Turks and Caicos Islands": "Cuba"}, inplace=True)
df_shape_id.replace({"Vatican City": "Italy"}, inplace=True)
df_shape_array.replace({"Vatican City": "Italy"}, inplace=True)
df_shape_id.replace({"Wallis and Futuna": "Fiji"}, inplace=True)
df_shape_array.replace({"Wallis and Futuna": "Fiji"}, inplace=True)
df_shape_id.replace({"Western Sahara": "Morocco"}, inplace=True)
df_shape_array.replace({"Western Sahara": "Morocco"}, inplace=True)

# variables
measures = {'mort': 1, 'yld': 3, 'yll': 4}
causes = {'all': 294, 'lri': 322, 'ncd': 409, 'lc': 426, 'ihd': 493, 'str': 494, 'copd': 509, 'cata': 671, 'diab': 976}
sexes = {'both': 3, 'male': 1, 'female': 2}
uncertainties = ['mean', 'upper', 'lower']
ages = {'eneo': 2, 'lneo': 3, 'pneo': 4, '1_4': 5, '5_9': 6, '10_14': 7, '15_19': 8, '20_24': 9, '25_29': 10, '30_34': 11, '35_39': 12, '40_44': 13, '45_49': 14, '50_54': 15, '55_59': 16, '60_64': 17, '65_69': 18, '70_74': 19, '75_79': 20, '80up': 21, 'all': 22, '80_84': 30, '85_89': 31, '90_94': 32, '95up': 235}

if process_population:
    ds_population = xr.Dataset()

    df_pop = pd.read_csv(f'{data_path}/GBD2019_pop_{year}.csv')

    df_pop.replace({"Cote d'Ivoire": "Ivory Coast"}, inplace=True)
    df_pop.replace({"Russian Federation": "Russia"}, inplace=True)
    df_pop.replace({"Congo": "Congo"}, inplace=True)
    df_pop = df_pop[df_pop.location_id != 533] # Remove the US state of Georgia as interfering with the country Georgia

    for uncertainty in uncertainties:
        for age, age_id in ages.items():
            for sex, sex_id in sexes.items():
                population_variable = f'age_fraction_{uncertainty}_{age}_{sex}'
                print(population_variable)

                for country_name in country_names:
                    print(f'{country_name}')
                    result_age = df_pop.loc[df_pop['location_name'] == country_name].loc[df_pop['age_group_id'] == age_id].loc[df_pop['year_id'] == year].loc[df_pop['sex_id'] == sex_id]['val']
                    result_all = df_pop.loc[df_pop['location_name'] == country_name].loc[df_pop['age_group_id'] == 22].loc[df_pop['year_id'] == year].loc[df_pop['sex_id'] == sex_id]['val']

                    if result_age.size > 0: # only if have results for this country
                        df_shape_array.loc[df_shape_array['country_name'] == country_name, population_variable] = float(result_age) / float(result_all)

                        pop_data = {}
                        for index, column in enumerate(df_shape_array.columns[:]):
                            pop_data[column] = df_shape_array[df_shape_array.columns[index]].values

                        # create 2d array from 3 1d columns of lon, lat and data (fill with nan and then the data)
                        pop_lat_vals, pop_lat_idx = np.unique(pop_data['lat'], return_inverse=True)
                        pop_lon_vals, pop_lon_idx = np.unique(pop_data['lon'], return_inverse=True)
                        pop_array = np.empty(pop_lat_vals.shape + pop_lon_vals.shape)
                        pop_array.fill(np.nan)
                        pop_array[pop_lat_idx, pop_lon_idx] = pop_data[population_variable]
                        ds_population[population_variable] = xr.DataArray(
                            pop_array,
                            dims=('lat', 'lon'),
                            coords={
                                'lat': pop_lat_vals,
                                'lon': pop_lon_vals
                            }
                        )


    ds_population.to_netcdf(f'{data_path}/GBD2019_population_{year}_{res}deg.nc')

if process_baseline_mortality:
    for measure, measure_id in measures.items():
        df_bm = pd.read_csv(f'{data_path}/GBD2019_BM_{year}_countries_cause_{measure.upper()}_rate_LRI-LC-COPD-IHD-DIAB-STR-NCD-ALL-CATA.csv')

        df_bm.replace({"Cote d'Ivoire": "Ivory Coast"}, inplace=True)
        df_bm.replace({"Russian Federation": "Russia"}, inplace=True)
        df_bm.replace({"Congo": "Congo"}, inplace=True)
        df_bm = df_bm.loc[df_bm.location_id != 533] # Remove the US state of Georgia as interfering with the country Georgia

        for cause, cause_id in causes.items():
            for sex, sex_id in sexes.items():
                for uncertainty in uncertainties:
                    ds_baseline_mortality = xr.Dataset() # one for each measure, cause, sex, and uncertainty

                    for age, age_id in ages.items():

                        if cause in ['lc', 'ihd', 'str  ', 'copd', 'cata', 'diab'] and age in ['eneo', 'lneo', 'pneo', '1_4', '5_9', '10_14', '15_19', '20_24']:
                            continue

                        if cause in ['cata'] and measure in ['mort', 'yll']:
                            continue

                        baseline_mortality_variable = f'i_{measure}_{cause}_{sex}_{uncertainty}_{age}'
                        print(baseline_mortality_variable)

                        for country_name in country_names:
                            print(f'{country_name}')
                            result = df_bm.loc[df_bm['location_name'] == country_name].loc[df_bm['measure_id'] == measure_id].loc[df_bm['sex_id'] == sex_id].loc[df_bm['age_id'] == age_id].loc[df_bm['cause_id'] == cause_id].loc[df_bm['year'] == year]['val']

                            if result.size > 0: # only if have results for this country
                                df_shape_array.loc[df_shape_array['country_name'] == country_name, baseline_mortality_variable] = float(result) / 100000

                                bm_data = {}
                                for index, column in enumerate(df_shape_array.columns[:]):
                                    bm_data[column] = df_shape_array[df_shape_array.columns[index]].values

                                # create 2d array from 3 1d columns of lon, lat and data (fill with nan and then the data)
                                bm_lat_vals, bm_lat_idx = np.unique(bm_data['lat'], return_inverse=True)
                                bm_lon_vals, bm_lon_idx = np.unique(bm_data['lon'], return_inverse=True)
                                bm_array = np.empty(bm_lat_vals.shape + bm_lon_vals.shape)
                                bm_array.fill(np.nan)
                                bm_array[bm_lat_idx, bm_lon_idx] = bm_data[baseline_mortality_variable]
                                ds_baseline_mortality[baseline_mortality_variable] = xr.DataArray(
                                    bm_array,
                                    dims=('lat', 'lon'),
                                    coords={
                                        'lat': bm_lat_vals,
                                        'lon': bm_lon_vals
                                    }
                                )

                    ds_baseline_mortality.to_netcdf(f'{data_path}/GBD2019_baseline_mortality_{measure}_{cause}_{sex}_{uncertainty}_{year}_{res}deg.nc') # for all ages and countries


