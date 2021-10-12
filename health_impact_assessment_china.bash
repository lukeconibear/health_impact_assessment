#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=01:00:00
#$ -pe smp 1
#$ -l h_vmem=64G

conda activate pangeo_latest
python health_impact_assessment_china.py
