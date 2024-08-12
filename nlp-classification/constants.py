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
SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v5.sif')  # singularity container path
# -------------------

# ----- PHYSICS -----
PHYSICS_SOURCE = '/usr/physics/python/Anaconda3-2022.10/etc/profile.d/conda.sh'
PHYSICS_CONDA = 'frac_attn' if 'chqu7424' in RT else '~/conda'
# -------------------

# ----- FUDAN-BRAIN -----
FUDAN_CONDA = 'frac_attn'
# -------------------

# ----- MODELS -----

MODEL_PREFIXES = ['fns', 'opfns', 'spfns', 'spopfns', 'rdfns', 'rdopfns', 'sink', 'dp']
MODEL_SUFFIX = 'former'

MODEL_NAMES = []
NAMES_DICT = {}
for MODEL_PREFIX in MODEL_PREFIXES:
    MODEL_NAMES.append(MODEL_PREFIX + MODEL_SUFFIX)
    NAMES_DICT[MODEL_PREFIX + MODEL_SUFFIX] = MODEL_PREFIX.upper()

NAMES_DICT.update({'imdb': 'IMDb', 'rotten_tomatoes': 'Rotten Tomatoes',
                   'eval_loss': 'Loss', 'eval_accuracy': 'Accuracy', 'eval_f1_score': r'$F_1$ score'}
                   )

# NAMES_DICT = {'fnsformer': 'FNS', 
#               'spfnsformer': 'SPFNS', 'spopfnsformer': 'SPOPFNS',               
#               'rdfnsformer': 'RDFNS', 'rdopfnsformer': 'RDOPFNS',
#               'sinkformer': 'SINK',
#               'dpformer': 'DP',
#               'imdb': 'IMDb', 'rotten_tomatoes': 'Rotten Tomatoes',
#               'eval_loss': 'Loss', 'eval_accuracy': 'Accuracy', 'eval_f1_score': r'$F_1$ score'}

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

LINESTYLE_DICT = {'spfnsformer': 'solid', 'spopfnsformer': 'solid',  # (0,(5,1))               
                  'rdfnsformer': 'solid', 'rdopfnsformer': 'solid',
                  'sinkformer': (0,(5,5)),
                  'dpformer': (0,(1,1))}

# NAMES_DICT = {'fnsformer': 'FNS', 
#               'spfnsformer': rf'FNS ($\mathbb{S}^{{d-1}}$)', 'spopfnsformer': rf'OPFNS ($\mathbb{S}^{{d-1}}$)',               
#               'rdfnsformer': rf'FNS ($\mathbb{R}^{{d}}$)', 'rdopfnsformer': rf'OPFNS ($\mathbb{R}^{{d}}$)',
#               'sinkformer': 'SINK',
#               'dpformer': 'DP',
#               'imdb': 'IMDb', 'rotten_tomatoes': 'Rotten Tomatoes',
#               'eval_loss': 'Loss', 'eval_accuracy': 'Accuracy', 'eval_f1_score': r'$F_1$ score'}              