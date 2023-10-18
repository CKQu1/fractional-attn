#!/bin/bash
#PBS -P dyson
##PBS -l select=1:ncpus=0:ngpus=2:mem=8gb
##PBS -l select=1:ncpus=0:ngpus=1:mem=6gb
#PBS -l select=1:ncpus=2:ngpus=0:mem=8gb
##PBS -l walltime=47:59:59
#PBS -l walltime=23:59:59
##PBS -l walltime=0:03:00
#PBS -e PBSout/
#PBS -o PBSout/

module load singularity
PBS_O_WORKDIR="/project/frac_attn/fractional-attn"
cd ${PBS_O_WORKDIR}

# check if relevant modules exist
#singularity exec --home ${PBS_O_WORKDIR}  ../built_containers/FaContainer_v2.sif python -c "import torch, transformers, dgl; print([dgl.__version__, transformers.__version__, torch.__version__,torch.cuda.is_available()])"

# with GPU
#singularity exec --nv --home ${PBS_O_WORKDIR} ../built_containers/FaContainer_v2.sif python train_classification_imdb.py

# without DDP
#singularity exec --home ${PBS_O_WORKDIR} ../built_containers/FaContainer_v2.sif python train_classification_imdb.py

# with DDP (GPU)
#export CUDA_VISIBLE_DEVICES=0,1
#singularity exec --home ${PBS_O_WORKDIR} ../built_containers/FaContainer_v2.sif CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_classification_imdb.py
#CUDA_VISIBLE_DEVICES=0,1 singularity exec --nv --home ${PBS_O_WORKDIR} ../built_containers/FaContainer_v2.sif python -m torch.distributed.launch --nproc_per_node=2 train_classification_imdb.py

# with DDP (GPU)
#export CUDA_VISIBLE_DEVICES=0,1
#CUDA_VISIBLE_DEVICES=0,1 singularity exec --home ${PBS_O_WORKDIR} ../built_containers/FaContainer_v2.sif python -m torch.distributed.launch --nproc_per_node=2 train_classification_imdb.py

# with DDP (CPU, seems to work)
#CUDA_VISIBLE_DEVICES=0 singularity exec --home ${PBS_O_WORKDIR} ../built_containers/FaContainer_v2.sif python -m torch.distributed.launch train_classification_imdb.py --ddp_backend ccl



