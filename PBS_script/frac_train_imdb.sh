#!/bin/bash
#PBS -P dyson
##PBS -l select=1:ncpus=0:ngpus=2:mem=8gb
##PBS -l select=1:ncpus=0:ngpus=1:mem=6gb
#PBS -l select=1:ncpus=4:ngpus=0:mem=20gb
##PBS -l walltime=47:59:59
#PBS -l walltime=5:59:59
##PBS -l walltime=0:03:00
#PBS -e PBSout/
#PBS -o PBSout/

module load singularity
PBS_O_WORKDIR="/project/frac_attn/fractional-attn"
cpath="../built_containers/FaContainer_v2.sif"
cd ${PBS_O_WORKDIR}

# with GPU
#singularity exec --nv --home ${PBS_O_WORKDIR} ${cpath} python frac_train_classification_imdb.py

# single CPU/GPU without DDP
#singularity exec --home ${PBS_O_WORKDIR} ${cpath} python frac_train_classification_imdb.py

# GPU with DDP
#export CUDA_VISIBLE_DEVICES=0,1
#singularity exec --home ${PBS_O_WORKDIR} ${cpath} CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 frac_train_classification_imdb.py
# or
#CUDA_VISIBLE_DEVICES=0,1 singularity exec --nv --home ${PBS_O_WORKDIR} ${cpath} python -m torch.distributed.launch --nproc_per_node=2 frac_train_classification_imdb.py

# CPU with DDP
singularity exec --home ${PBS_O_WORKDIR} ${cpath} torchrun --nproc_per_node=4 frac_train_classification_imdb.py
# or
#singularity exec --home ${PBS_O_WORKDIR} ${cpath} python -m torch.distributed.launch --nproc_per_node=4 --use_env frac_train_classification_imdb.py run --backend=gloo

