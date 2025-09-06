#!/bin/bash
#PBS -N lra_dataset
#PBS -P uu69
#PBS -q copyq
#PBS -o /scratch/uu69/cq5024/projects/fractional-attn/long-range-arena/.PBSout -e /scratch/uu69/cq5024/projects/fractional-attn/long-range-arena/.PBSout
#PBS -l ncpus=1
#PBS -l mem=64GB
#PBS -l jobfs=60GB
#PBS -l storage=gdata/uu69+scratch/uu69
#PBS -l walltime=04:00:00
##PBS -l walltime=00:02:00
#PBS -l wd

# ----- CHECK PATH -----
# echo $PBS_JOBFS > .droot/jobfs_path.txt
# cd $PBS_JOBFS
# echo '----- PBS_JOBFS path -----'
# pwd
# echo '--------------------'

# ----- VERIFY -----
# echo $PBS_JOBFS > .droot/jobfs_path.txt
# source /scratch/uu69/cq5024/myenvs/fsa/bin/activate
# cd /scratch/uu69/cq5024/projects/fractional-attn/long-range-arena
# python3 _lra_preprocess.py

# ----- DOWNLOAD DATASET -----
echo $PBS_JOBFS > .droot/jobfs_path.txt
cd $PBS_JOBFS
wget -c https://storage.googleapis.com/long-range-arena/lra_release.gz -O - | tar -xz

cd /scratch/uu69/cq5024/projects/fractional-attn/long-range-arena
source /scratch/uu69/cq5024/myenvs/fsa/bin/activate

python3 _lra_preprocess.py  --name=aan
echo '---------- AAN complete! ----------'
python3 _lra_preprocess.py  --name=pathfinder
echo '---------- PATHFINDER complete! ----------'

exit