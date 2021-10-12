#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=01:00:00
#$ -pe ib 1
#$ -l h_vmem=128G

conda activate pangeo_latest
python health_impact_assessment_global.py
