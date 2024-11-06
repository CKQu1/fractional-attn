import matplotlib as mpl
import os
from mutils import njoin
from matplotlib.cm import get_cmap

# ----- GENERAL -----
RT = os.path.abspath(os.getcwd())
DROOT = njoin(RT, '.droot')
FIGS_DIR = njoin(DROOT, 'figs_dir')
SCRIPT_DIR = njoin(DROOT, 'submitted_scripts')

CLUSTER = 'ARTEMIS' if 'project' in DROOT else 'PHYSICS' if 'headnode' in DROOT else 'FUDAN_BRAIN'
# -------------------

# ----- ARTEMIS -----
PROJECTS = ['phys_DL','PDLAI','dnn_maths','dyson','vortex_dl','frac_attn', 'ddl']
BPATH = njoin('/project')  # path for binding to singularity container
#SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v5.sif')  # singularity container path
SPATH = njoin('/project/frac_attn/built_containers/pydl.img')
# -------------------

# ----- PHYSICS -----
PHYSICS_SOURCE = '/usr/physics/python/Anaconda3-2022.10/etc/profile.d/conda.sh'
PHYSICS_CONDA = 'frac_attn' if 'chqu7424' in RT else '~/conda'
# -------------------

# ----- FUDAN-BRAIN -----
FUDAN_CONDA = 'frac_attn'
# -------------------

# ----- MODELS -----

MODEL_PREFIXES = ['fns', 'opfns', 'spfns', 'spopfns', 'rdfns', 'rdopfns', 'sink', 'opsink', 'dp', 'opdp']
MODEL_SUFFIX = 'vit'

MODEL_NAMES = []
NAMES_DICT = {}
for MODEL_PREFIX in MODEL_PREFIXES:
    MODEL_NAMES.append(MODEL_PREFIX + MODEL_SUFFIX)
    NAMES_DICT[MODEL_PREFIX + MODEL_SUFFIX] = MODEL_PREFIX.upper()

NAMES_DICT.update({'cifar10': 'CIFAR10', 'val_loss': 'Eval Loss',
                   'val_acc': 'Eval Acc.', 'train_loss': 'Train Loss', 'train_acc': 'Train Acc.'}
                   )               

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

LINESTYLE_DICT = {'spfns'+MODEL_SUFFIX: 'solid', 'spopfns'+MODEL_SUFFIX: 'solid',  # (0,(5,1))               
                  'rdfns'+MODEL_SUFFIX: 'solid', 'rdopfns'+MODEL_SUFFIX: 'solid',
                  'sink'+MODEL_SUFFIX: (0,(5,5)), 'opsink'+MODEL_SUFFIX: (0,(5,5)),
                  'dp'+MODEL_SUFFIX: (0,(1,1)), 'opdp'+MODEL_SUFFIX: (0,(1,1))
                  }        