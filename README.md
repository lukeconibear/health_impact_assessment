#### Code for health impact assessments
##### Functions
- Import numpy arrays (`import_npz.py`).  
- Find index within array nearest to a given value (`find_nearest.py`).  
- Create the ozone metric used for health impact assessments (`create_o3_metric.py`).  
- GEMM NCD+LRI for long-term PM2.5 exposure (`calc_hia_gemm_ncdlri.py`).  
    - Variant using Numba (`calc_hia_gemm_ncdlri_numba.py`).  
- GEMM 5COD for long-term PM2.5 exposure (`calc_hia_gemm_5cod.py`).  
- GBD2017 for long-term PM2.5 exposure (`calc_hia_gbd2017_pm25.py`).  
- GBD2017 for long-term O3 exposure (`calc_hia_gbd2017_o3.py`).  
- Cut an array by a shapefile (`cutshapefile.py`).  
- Cut the health impact assessment arrays by shapefiles (`shapefile_hia.py`).  

##### Examples
- Regridding to a global grid (`example_regridding.py`).  
- Application of cutting an array by a shapefile (`example_crop_array_to_shapefile.py`).  
- Health impact assessment for long-term PM2.5 and O3 exposures using GBD2017 exposure-outcome associations (`example_hia_gbd2017.py`).  
- Health impact assessment for long-term PM2.5 exposure using the GEMM NCD+LRI exposure-outcome associations (`example_hia_gbd2017.py`).  
- Cropping health impact assessments by shapefile (`example_hia_split_shapefile.py`).  
