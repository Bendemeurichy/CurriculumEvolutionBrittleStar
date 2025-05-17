#!/bin/bash

#PBS -N brittlestar
#PBS -l nodes=1:ppn=4
#PBS -l walltime=6:00:00
#PBS -l mem=24gb
#PBS -m abe

source .bashrc

cd $VSC_DATA

conda activate biorobot

cd brittlestar/NEAT

# Run the Python script

python -u train.py --mode curriculum

# python -u train.py --mode no_curriculum


