#!/bin/bash
#PBS -P PROJECT
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=23:59:59

module load singularity

cd "PBS_O_WORKDIR"
singularity exec CondaContainer.simg python train_classification_imdb.py 
