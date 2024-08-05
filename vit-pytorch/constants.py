import matplotlib as mpl
import os
from mutils import njoin
from matplotlib.cm import get_cmap

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

SCRIPT_DIR = njoin(DROOT, 'submitted_scripts')

# -------------------

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

# color for model hyperparameters
#HYP_CM = 'gist_ncar'
HYP_CM = 'turbo'
HYP_CMAP = get_cmap(HYP_CM)
HYP_CNORM = mpl.colors.Normalize(vmin=1, vmax=2)

def HYP_TRANS(alpha):
    min_trans, max_trans = 0.5, 1
    min_alpha, max_alpha = 1, 2
    m = (min_trans - max_trans) / (max_alpha - min_alpha)
    b = max_trans - m * min_alpha
    return m*alpha + b

LINESTYLE_DICT = {'spfnsvit': 'solid', 'spopfnsvit': (0,(5,1)),               
                  'rdfnsvit': 'solid', 'rdopfnsvit': (0,(5,1)),
                  'sinkvit': (0,(1,1)),
                  'dpvit': (0,(1,1))}              