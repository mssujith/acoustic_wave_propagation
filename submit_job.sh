#!/bin/bash

# #PBS -l walltime=01:00:00
#PBS -l nodes=4:ppn=16
#PBS -N test

cd /home/mssujith/GITHUB/acoustic_wave_propagation/
python forward_model_2d.py


