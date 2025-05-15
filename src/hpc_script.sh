#!/bin/bash

#PBS -N brittlestar
#PBS -l nodes=1:ppn=4:gpus=1
#PBS -l walltime=12:00:00
#PBS -l mem=32gb
#PBS -m abe

cd $VSC_DATA

conda activate biorobot

cd brittlestar/NEAT

# Run the Python script

python train.py

