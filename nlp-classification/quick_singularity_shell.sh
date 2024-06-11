#!/bin/sh
PBS_O_WORKDIR="/project/frac_attn/fractional-attn" 
cpath="../../built_containers/FaContainer_v4.sif" 
bpath1="/project"
bpath2="/etc/localtime"
#singularity shell -B ${bpath} --home ${PBS_O_WORKDIR} ${cpath}
singularity shell -B ${bpath1} ${cpath}