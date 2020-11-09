#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=48:00:00
#$ -pe ib 1
#$ -l h_vmem=192G

conda activate python3

python create_arrays_gbd2019_baseline_mortality_population.py
