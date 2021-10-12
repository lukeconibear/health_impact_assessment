#### Code for health impact assessments from air pollution exposure
##### Functions
- Import numpy arrays (`import_npz.py`).  
- Find index within array nearest to a given value (`find_nearest.py`).  
- Create the ozone metric used for health impact assessments (`create_o3_metric.py`).  
- GEMM NCD+LRI for long-term PM2.5 exposure (`calc_hia_gemm_ncdlri.py`).  
    - Variant using Numba (`calc_hia_gemm_ncdlri_numba.py`).  
- GEMM 5COD for long-term PM2.5 exposure (`calc_hia_gemm_5cod.py`).  
- GBD2017/2019 for long-term PM2.5 exposure (`calc_hia_gbd2017_pm25.py`, `calc_hia_gbd2019_pm25.py`).  
- GBD2017/2019 for long-term O3 exposure (`calc_hia_gbd2017_o3.py`, `calc_hia_gbd2019_o3.py`).  
- Create global arrays of GBD2019 baseline mortality and population age (`create_arrays_gbd2019_baseline_mortality_population.py/bash`).  
- Test calculation for GBD2019 (`hia_gbd2019.py/bash`).  
- Cut an array by a shapefile (`cutshapefile.py`).  
- Cut the health impact assessment arrays by shapefiles (`shapefile_hia.py`).  

##### Examples
- Regridding to a global grid (`example_regridding.py`).  
- Application of cutting an array by a shapefile (`example_crop_array_to_shapefile.ipynb`).  
- Plotting concentrations and calculating exposure over an area (`example_popweighted_exposure.ipynb`).
- Health impact assessment for long-term PM2.5 and O3 exposures using GBD2017 exposure-outcome associations (`example_hia_gbd2017.py`).  
- Health impact assessment for long-term PM2.5 exposure using the GEMM NCD+LRI exposure-outcome associations (`example_hia_gbd2017.py`, `health_impact_assessment_china.py`, `health_impact_assessment_global.py`).  
- Cropping health impact assessments by shapefile (`example_hia_split_shapefile.py`).  
- Health impact assessment for short-term PM2.5 exposure (`example_hia_shortterm.py`).  

