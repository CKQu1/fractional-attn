import os
from mutils import njoin

# ----- ARTEMIS -----
PROJECTS = ['phys_DL','PDLAI','dnn_maths','dyson','vortex_dl','frac_attn', 'ddl']
# -------------------

# ----- PHYSICS -----
PHYSICS_SOURCE = '/usr/physics/python/Anaconda3-2022.10/etc/profile.d/conda.sh'
PHYSICS_CONDA = 'frac_attn'
# -------------------

# ----- PATHS -----
RT = os.path.abspath(os.getcwd())
if 'project' in RT:
    DROOT = njoin('/project/frac_attn/fractional-attn/vit-pytorch', '.droot')
else:
    DROOT = njoin(RT, '.droot')

FIGS_DIR = njoin(DROOT, 'figs_dir')

BPATH = njoin('/project')  # path for binding to singularity container
SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v5.sif')  # singularity container path

MODEL_NAMES = ['fnsvit', 'opfnsvit', 'sinkvit', 'dpvit',
               'spfnsvit', 'spopfnsvit',
               'rdfnsvit', 'rdopfnsvit']  # model str names
               
NAMES_DICT = {'fnsvit': 'FNS', 'opfnsvit': 'OPFNS',
              'spfnsvit': 'SPDMFNS', 'spopfnsvit': 'SPOPFNS',
              'rdfnsvit': 'RDFNS', 'rdopfnsvit': 'RDOPFNS',
              'sinkvit': 'SINK', 'dpvit': 'DP', 
              'cifar10': 'CIFAR10',
              'val_loss': 'Eval Loss', 'val_acc': 'Eval Acc.', 'train_loss': 'Train Loss', 'train_acc': 'Train Acc.'
              }