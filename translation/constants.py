import os
from mutils import njoin

PROJECTS = ['phys_DL','PDLAI','dnn_maths','dyson','vortex_dl','frac_attn', 'ddl']

#DROOT = njoin(os.path.abspath(os.getcwd()), 'droot')
DROOT = njoin('/project/frac_attn/fractional-attn/translation', '.droot')
FIGS_DIR = njoin(DROOT, 'figs_dir')

BPATH = njoin('/project')  # path for binding to singularity container
SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v4.sif')  # singularity container path

MODEL_NAMES = ['fnsnmt', 'opfnsnmt', 
               'rdfnsnmt', 'rdopfnsnmt',
               'sinknmt', 'dpnmt']  # model str names
NAMES_DICT = {'fnsnmt': 'FNS', 'opfnsnmt': 'OPFNS', 
              'rdfnsnmt': 'RDFNS', 'rdopfnsnmt': 'RDOPFNS',
              'sinknmt': 'SINK',
              'dpnmt': 'DP',
              'iwslt14': 'IWSLT14',
              'train_loss': 'Train loss', 'val_loss': 'Val loss', 'val_bleu': 'Bleu', 
              'eval_f1_score': r'$F_1$ score'
              }