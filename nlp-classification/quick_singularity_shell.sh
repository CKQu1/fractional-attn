#!/bin/sh
PBS_O_WORKDIR="/project/frac_attn/fractional-attn" 
#cpath="../../built_containers/FaContainer_v5.sif" 
cpath="../../built_containers/pydl.img"
bpath1="/project"
bpath2="/usr/bin/nvidia-smi"
#singularity shell -B ${bpath} --home ${PBS_O_WORKDIR} ${cpath}
#singularity shell --bind ${bpath1},${bpath2} ${cpath}
singularity shell --nv --bind ${bpath1} ${cpath}