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

# ----- TASKS -----
TASKS = ["lra-listops", "lra-retrieval", "lra-text", "lra-pathfinder32-curv_contour_length_14", "lra-image"]
# -------------------

# ----- MODELS -----
MODEL_PREFIXES = ['fns', 'opfns', 'spfns', 'spopfns', 'rdfns', 'rdopfns', 'sink', 'softmax']
MODEL_PREFIXES_UPPER = [model_prefix.upper() for model_prefix in MODEL_PREFIXES]
MODEL_SUFFIX = 'former'

MODEL_NAMES = []
NAMES_DICT = {}
for MODEL_PREFIX in MODEL_PREFIXES:
    if MODEL_PREFIX != 'softmax':
        MODEL_NAMES.append(MODEL_PREFIX + MODEL_SUFFIX)
        NAMES_DICT[MODEL_PREFIX] = MODEL_PREFIX.upper()
    else:
        MODEL_NAMES.append('dp' + MODEL_SUFFIX)
        NAMES_DICT[MODEL_PREFIX] = 'dp'.upper()        

NAMES_DICT.update({'imdb': 'IMDb', 'rotten_tomatoes': 'Rotten Tomatoes',
                   'eval_loss': 'Loss', 'eval_accuracy': 'Accuracy', 'eval_f1_score': r'$F_1$ score'}
                   )
# -------------------

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

OTHER_COLORS = ['m', 'dimgray']
OTHER_COLORS_DICT = {'sink': OTHER_COLORS[0], 'softmax': OTHER_COLORS[1]}

LINESTYLE_DICT = {'spfns': 'solid', 'spopfns': 'solid',  # (0,(5,1))               
                  'rdfns': 'solid', 'rdopfns': 'solid',
                  'sink': (0,(5,5)),
                  'softmax': (0,(1,1))}            