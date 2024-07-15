import os
from mutils import njoin

PROJECTS = ['phys_DL','PDLAI','dnn_maths','dyson','vortex_dl','frac_attn', 'ddl']

RT = os.path.abspath(os.getcwd())
if 'project' in RT:
    DROOT = njoin('/project/frac_attn/fractional-attn/long-range-arena', '.droot')
else:
    DROOT = njoin(RT, 'long-range-arena', '.droot')

FIGS_DIR = njoin(DROOT, 'figs_dir')

BPATH = njoin('/project')  # path for binding to singularity container
SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v5.sif')  # singularity container path

DATASET_NAMES = ['imdb-classification', 'lra-cifar-classification',
                 'listops-classification', 'aan-classification',
                 'pathfinder-classification', 'pathx-classification'] 

MODEL_NAMES = ['fnsformer', 'opfnsformer', 'sinkformer', 'dpformer']  # model str names
NAMES_DICT = {'fnsformer': 'FNS', 'opfnsformer': 'OPFNS', 
              'sinkformer': 'SINK',
              'dpformer': 'DP',
              'iwslt14': 'IWSLT14',
              'train_loss': 'Train loss', 'val_loss': 'Val loss', 'val_bleu': 'Bleu', 
              'eval_f1_score': r'$F_1$ score'
              }