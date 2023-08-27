#!/bin/bash
#PBS -P phys_DL
PBS_O_WORKDIR="/project/frac_attn/Diffuser"
cd ${PBS_O_WORKDIR}
source PBS_script/project_rotation.sh
cur_project="$(get_project)"
#PBS -P $cur_project
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb
##PBS -l select=1:ncpus=1:ngpus=0:mem=4gb
#PBS -l walltime=2:00:00
##PBS -l walltime=23:59:59
#PBS -e PBSout/
#PBS -o PBSout/

singularity exec ../built_containers/FaContainer.sif python train_classification_imdb.py