#!/usr/bin/env python3
import os
import re
import time
import sys
import glob
import joblib
import xarray as xr
import numpy as np
import dask.bag as db
import geopandas as gpd
import pandas as pd
from dask_jobqueue import SGECluster
from dask.distributed import Client
from numba import njit, typeof, typed, types, jit
from functions_health_impact_assessment import (shapefile_hia, dict_to_typed_dict,
                                                outcome_per_age_ncdlri, outcome_total,
                                                dalys_age, dalys_total,
                                                rates_total, calc_hia_gemm_ncdlri,
                                                create_attribute_fraction, calc_hia_gbd2017_o3,
                                                health_impact_assessment_pm25, health_impact_assessment_o3,
                                                clips, dict_pops, pop_xx, pop_yy, dict_ages,
                                                dict_bm, dict_af, dict_gemm)

output = "PM2_5_DRY"
#output = "o3_6mDM8h"

def main():
    # dask cluster and client
    n_jobs = 20    
    n_processes = 1
    n_workers = n_processes * n_jobs

    cluster = SGECluster(
        interface="ib0",
        walltime="02:00:00",
        memory=f"48 G",
        resource_spec=f"h_vmem=48G",
        scheduler_options={
            "dashboard_address": ":7777",
        },
        job_extra=[
            "-cwd",
            "-V",
            f"-pe smp {n_processes}",
            f"-l disk=48G",
        ],
        local_directory=os.sep.join([os.environ.get("PWD"), "dask-hia-space"]),
    )

    client = Client(cluster)
    cluster.scale(jobs=n_jobs)
    time_start = time.time()

    # dask bag and process
    simulations = [f'emulator_Base_CLE_2020_{output}']
    
    #simulations = []
    #simulations.append(f'wrfchem_Base_CLE_2020_{output}')
    #simulations.append(f'wrfchem_Base_CLE_2050_{output}')
    #simulations.append(f'wrfchem_Base_MFR_2050_{output}')
    #simulations.append(f'wrfchem_SDS_MFR_2050_{output}')

    #for year in ['2020', '2030', '2040', '2050']:
    #    for scenario in ['Base_CLE', 'Base_MFR', 'SDS_MFR']:        
    #        for sim in ['', '_RES', '_IND', '_TRA', '_AGR', '_ENE', '_NO_RES', '_NO_IND', '_NO_TRA', '_NO_AGR', '_NO_ENE']:
    #            simulations.append(f'emulator_{scenario}_{year}{sim}_{output}')

    print(f"predicting for {len(simulations)} custom outputs ...")
    bag_simulations = db.from_sequence(simulations, npartitions=n_workers)

    if output == "PM2_5_DRY":
        bag_simulations.map(health_impact_assessment_pm25).compute()
    elif output == "o3_6mDM8h":
        bag_simulations.map(health_impact_assessment_o3).compute()

    time_end = time.time() - time_start
    print(f"completed in {time_end:0.2f} seconds, or {time_end / 60:0.2f} minutes, or {time_end / 3600:0.2f} hours")

    client.close()
    cluster.close()


if __name__ == "__main__":
    main()

