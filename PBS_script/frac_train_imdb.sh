#!/bin/bash
#PBS -P frac_attn
#PBS -l select=1:ncpus=0:ngpus=1:mem=16gb
##PBS -l select=1:ncpus=4:ngpus=0:mem=8gb
##PBS -l walltime=2:00:00
#PBS -l walltime=23:59:59
#PBS -e PBSout/
#PBS -o PBSout/

PBS_O_WORKDIR="/project/frac_attn/fractional-attn"
cd ${PBS_O_WORKDIR}
singularity exec ../built_containers/FaContainer.sif python frac_train_classification_imdb.py

