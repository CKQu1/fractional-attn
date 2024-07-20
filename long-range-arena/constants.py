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
    DROOT = njoin('/project/frac_attn/fractional-attn/long-range-arena', '.droot')
else:
    DROOT = njoin(RT, '.droot')

FIGS_DIR = njoin(DROOT, 'figs_dir')

BPATH = njoin('/project')  # path for binding to singularity container
SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v5.sif')  # singularity container path

# ----- TRAINING -----

MODEL_NAMES = ['fnsformer', 'opfnsformer', 'sinkformer', 'dpformer']  # model str names
NAMES_DICT = {'fnsformer': 'FNS', 'opfnsformer': 'OPFNS', 
              'sinkformer': 'SINK',
              'dpformer': 'DP',
              'iwslt14': 'IWSLT14',
              'train_loss': 'Train loss', 'val_loss': 'Val loss', 'val_bleu': 'Bleu', 
              'eval_f1_score': r'$F_1$ score'
              }

DATASET_NAMES = ['imdb-classification', 'lra-cifar-classification',
                 'listops-classification', 'aan-classification',
                 'pathfinder-classification', 'pathx-classification']     

DATASET_EPOCHS = {'imdb-classification':5, 'lra-cifar-classification':100,
                  'listops-classification':5, 'aan-classification':5,
                  'pathfinder-classification':5, 'pathx-classification':5}                

DATASET_STEPS = {'imdb-classification':100000, 'lra-cifar-classification':100000,
                  'listops-classification':500000, 'aan-classification':500000,
                  'pathfinder-classification':500000, 'pathx-classification':500000}                                