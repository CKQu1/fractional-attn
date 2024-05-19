import os
from mutils import njoin

PROJECTS = ['phys_DL','PDLAI','dnn_maths','dyson','vortex_dl','frac_attn']

#DROOT = njoin(os.path.abspath(os.getcwd()), 'droot')
DROOT = njoin('/project/frac_attn/fractional-attn/vit-pytorch', '.droot')
FIGS_DIR = njoin(DROOT, 'figs_dir')

BPATH = njoin('/project')  # path for binding to singularity container
SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v3.sif')  # singularity container path

MODEL_NAMES = ['fnsvit', 'dpvit']  # model str names
NAMES_DICT = {'fnsvit': 'FNS', 'dpvit': 'DP',
              'cifar10': 'Cifar10',
              'val_loss': 'Eval Loss', 'val_acc': 'Eval Acc.', 'train_loss': 'Train Loss', 'train_acc': 'Train Acc.'
              }