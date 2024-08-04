import matplotlib as mpl
import os
from mutils import njoin
from matplotlib.cm import get_cmap

# ----- ARTEMIS -----
PROJECTS = ['phys_DL','PDLAI','dnn_maths','dyson','vortex_dl','frac_attn', 'ddl']
# -------------------

# ----- PATHS -----
RT = os.path.abspath(os.getcwd())
if 'project' in RT:
    DROOT = njoin('/project/frac_attn/fractional-attn/nlp-classification', '.droot')
else:
    DROOT = njoin(RT, '.droot')

FIGS_DIR = njoin(DROOT, 'figs_dir')

BPATH = njoin('/project')  # path for binding to singularity container
SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v5.sif')  # singularity container path

SCRIPT_PATH = njoin(DROOT, 'submitted_scripts')

# -------------------

# ----- PHYSICS -----
PHYSICS_SOURCE = '/usr/physics/python/Anaconda3-2022.10/etc/profile.d/conda.sh'
PHYSICS_CONDA = 'frac_attn' if 'chqu7424' in RT else '~/conda'
# -------------------

# ----- FIGURES -----
MODEL_NAMES = ['fnsformer', 
               'spfnsformer','spopfnsformer',
               'rdfnsformer', 'rdopfnsformer',
               'sinkformer',
               'dpformer', 
               'l2former']  # model str names
               
NAMES_DICT = {'fnsformer': 'FNS', 
              'spfnsformer': 'SPFNS', 'spopfnsformer': 'SPOPFNS',               
              'rdfnsformer': 'RDFNS', 'rdopfnsformer': 'RDOPFNS',
              'sinkformer': 'SINK',
              'dpformer': 'DP',
              'imdb': 'IMDb', 'rotten_tomatoes': 'Rotten Tomatoes',
              'eval_loss': 'Loss', 'eval_accuracy': 'Accuracy', 'eval_f1_score': r'$F_1$ score'}



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

LINESTYLE_DICT = {'spfnsformer': 'solid', 'spopfnsformer': (0,(5,1)),               
                  'rdfnsformer': 'solid', 'rdopfnsformer': (0,(5,1)),
                  'sinkformer': (0,(5,5)),
                  'dpformer': (0,(5,10))}

# NAMES_DICT = {'fnsformer': 'FNS', 
#               'spfnsformer': rf'FNS ($\mathbb{S}^{{d-1}}$)', 'spopfnsformer': rf'OPFNS ($\mathbb{S}^{{d-1}}$)',               
#               'rdfnsformer': rf'FNS ($\mathbb{R}^{{d}}$)', 'rdopfnsformer': rf'OPFNS ($\mathbb{R}^{{d}}$)',
#               'sinkformer': 'SINK',
#               'dpformer': 'DP',
#               'imdb': 'IMDb', 'rotten_tomatoes': 'Rotten Tomatoes',
#               'eval_loss': 'Loss', 'eval_accuracy': 'Accuracy', 'eval_f1_score': r'$F_1$ score'}              