#!/usr/bin/env python3
import xarray as xr

def create_ozone_metric(o3_annual):
    '''
    Description:
        Create seasonal (maximum 6-month mean), daily maximum, 8-hour, O3 concentration (6mDM8h) for GBD2017

    Arguments:
        o3_annual (xarray.core.dataset.Dataset): Annual ozone to extract the metric from.

    Returns:
        o3_6mDM8h (xarray.core.dataset.Dataset): Ozone metric.
    '''
    # first: 24, 8-hour, rolling mean, O3 concentrations
    o3_6mDM8h_8hrrollingmean = o3_annual.rolling(time=8).construct('window').mean('window')

    # second: find the max of these each day (daily maximum, 8-hour)
    o3_6mDM8h_dailymax = o3_6mDM8h_8hrrollingmean.sortby('time').resample(time='24H').max().compute()

    # third: 6-month mean - to account for different times when seasonal maximums e.g. different hemispheres
    o3_6mDM8h_6monthmean = o3_6mDM8h_dailymax.resample(time='6M').mean()

    # fourth: maximum of these
    o3_6mDM8h = o3_6mDM8h_6monthmean.max(dim='time')

    return o3_6mDM8h
