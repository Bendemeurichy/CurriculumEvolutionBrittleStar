#!/bin/bash

#PBS -N brittlestar
#PBS -l nodes=1:ppn=1:gpus=2
#PBS -l walltime=01:00:00
#PBS -l mem=16gb
#PBS -m abe

module load Miniconda3/23.5.2-0

conda activate biorobot

# Set the working directory to the directory where the script is located
cd NEAT
# Run the Python script
python train.py

