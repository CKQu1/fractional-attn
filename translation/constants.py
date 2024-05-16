import os
from mutils import njoin

PROJECTS = ['phys_DL','PDLAI','dnn_maths','dyson','vortex_dl','frac_attn', 'ddl']

#DROOT = njoin(os.path.abspath(os.getcwd()), 'droot')
DROOT = njoin('/project/frac_attn/fractional-attn/vit-pytorch', '.droot')
FIGS_DIR = njoin(DROOT, 'figs_dir')

BPATH = njoin('/project')  # path for binding to singularity container
SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v3.sif')  # singularity container path

MODEL_NAMES = ['fnstranslation', 'dptranslation']  # model str names
NAMES_DICT = {'fnstranslation': 'FNS', 'dptranslation': 'DP',
              'iwslt14': 'IWSLT14',
              'eval_loss': 'Loss', 'eval_accuracy': 'Accuracy', 'eval_f1_score': r'$F_1$ score'
              }