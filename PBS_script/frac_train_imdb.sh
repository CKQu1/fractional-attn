#!/bin/bash
#PBS -P frac_attn
##PBS -l select=1:ncpus=0:ngpus=1:mem=24gb
##PBS -l select=1:ncpus=4:ngpus=0:mem=16gb
#PBS -l select=1:ncpus=4:ngpus=0:mem=12gb
##PBS -l walltime=2:00:00
##PBS -l walltime=47:59:59
#PBS -l walltime=23:59:59
#PBS -e PBSout/
#PBS -o PBSout/

module load singularity
PBS_O_WORKDIR="/project/frac_attn/fractional-attn"
cd ${PBS_O_WORKDIR}

# check if relevant modules exist
#singularity exec --home ${PBS_O_WORKDIR}  ../built_containers/FaContainer.sif python -c "import torch, transformers, dgl; print([dgl.__version__, transformers.__version__, torch.__version__,torch.cuda.is_available()])"

singularity exec --home ${PBS_O_WORKDIR} ../built_containers/FaContainer.sif python frac_train_classification_imdb.py
