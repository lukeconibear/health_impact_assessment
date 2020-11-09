#!/bin/bash -l
#$ -cwd -V
#$ -l h_rt=12:00:00
#$ -pe ib 1
#$ -l h_vmem=192G

conda activate python3

python hia_gbd2019.py
