import matplotlib as mpl
import os
from UTILS.mutils import njoin
from matplotlib.cm import get_cmap

# ----- GENERAL -----
RT = os.path.abspath(os.getcwd())
DROOT = njoin(RT, '.droot')
FIGS_DIR = njoin(DROOT, 'figs_dir')
SCRIPT_DIR = njoin(DROOT, 'submitted_scripts')

#CLUSTER = 'ARTEMIS' if 'project' in DROOT else 'PHYSICS' if 'taiji1' in DROOT else 'FUDAN_BRAIN'
if 'uu69' in DROOT:
    CLUSTER = 'GADI' 
elif 'taiji1' in DROOT:
    CLUSTER = 'PHYSICS'
else:
    CLUSTER = None
# -------------------

# ----- RESOURCES -----
RESOURCE_CONFIGS = {
    "GADI": {
        True:  {"q": "gpuvolta", "ngpus": 1, "ncpus": 12},
        False: {"q": "normal",   "ngpus": 0, "ncpus": 1},
    },
    "PHYSICS": {
        True:  {"q": "l40s", "ngpus": 1, "ncpus": 1},
        False: {"q": "taiji", "ngpus": 0, "ncpus": 1},
    },
}
# -------------------

# ----- GADI -----
GADI_PROJECTS = ['uu69']
GADI_SOURCE = '/scratch/uu69/cq5024/myenvs/fsa/bin/activate'
# -------------------

# ----- PHYSICS -----
PHYSICS_SOURCE = '/usr/physics/python/Anaconda3-2022.10/etc/profile.d/conda.sh'
PHYSICS_CONDA = 'frac_attn' if 'chqu7424' in RT else '~/conda'
# -------------------

# ----- FUDAN-BRAIN -----
FUDAN_CONDA = 'frac_attn'
# -------------------

# ----- ARTEMIS -----
ARTEMIS_PROJECTS = ['phys_DL','PDLAI','dnn_maths','dyson','vortex_dl','frac_attn', 'ddl']
BPATH = njoin('/project')  # path for binding to singularity container
#SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v5.sif')  # singularity container path
SPATH = njoin('/project/frac_attn/built_containers/pydl.img')
# -------------------

# ----- MODELS -----

GLOVE_DIMS = [50,100,200,300]

#MODEL_PREFIXES = ['fns', 'opfns', 'spfns', 'spopfns', 'rdfns', 'rdopfns', 'sink', 'opsink', 'dp', 'opdp']
MODEL_PREFIXES = ['fns', 'opfns', 'spfns', 'opspfns', 'rdfns', 'oprdfns', 
                  'v2_rdfns', 'opv2_rdfns', 'sink', 'opsink', 'dp', 'opdp']
MODEL_SUFFIX = 'former'

MODEL_NAMES = []
NAMES_DICT = {}
for MODEL_PREFIX in MODEL_PREFIXES:
    MODEL_NAMES.append(MODEL_PREFIX + MODEL_SUFFIX)

    if 'fns' in MODEL_PREFIX.lower():
        if 'rd' in MODEL_PREFIX.lower():
            if 'v2_' in MODEL_PREFIX.lower():
                NAMES_DICT[MODEL_PREFIX + MODEL_SUFFIX] = r'$\overline{{FRAC}}$ $(\mathbb{{R}}^d)$'
            else:
                NAMES_DICT[MODEL_PREFIX + MODEL_SUFFIX] = r'FRAC $(\mathbb{{R}}^d)$'
        else:                    
            NAMES_DICT[MODEL_PREFIX + MODEL_SUFFIX] = r'FRAC $(\mathcal{{S}}^{{d-1}})$'
    else:
        NAMES_DICT[MODEL_PREFIX + MODEL_SUFFIX] = MODEL_PREFIX.upper()    

NAMES_DICT.update({'imdb': 'IMDb', 'rotten_tomatoes': 'Rotten Tomatoes',
                   'train_loss': 'Train Loss', 'train_acc': 'Train Acc.',
                   'val_loss': 'Test Loss', 'val_acc': 'Test Acc.', 
                   'train_loss': 'Train Loss', 'train_acc': 'Train Acc.'}
                   )

# ----- COLORS -----

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
OTHER_COLORS_DICT = {'sink'+MODEL_SUFFIX: OTHER_COLORS[0], 'dp'+MODEL_SUFFIX: OTHER_COLORS[1],
                     'opsink'+MODEL_SUFFIX: OTHER_COLORS[0], 'opdp'+MODEL_SUFFIX: OTHER_COLORS[1]}

# ----- LINESTYLE_DICT -----

LINESTYLE_DICT = {'spfns'+MODEL_SUFFIX: 'solid', 'spopfns'+MODEL_SUFFIX: 'solid',  # (0,(5,1))               
                  'rdfns'+MODEL_SUFFIX: 'solid', 'rdopfns'+MODEL_SUFFIX: 'solid',
                  'opspfns'+MODEL_SUFFIX: 'solid', 'oprdfns'+MODEL_SUFFIX: 'solid', 
                  'v2_rdfns'+MODEL_SUFFIX: 'solid', 'opv2_rdfns'+MODEL_SUFFIX: 'solid',
                  #'sink'+MODEL_SUFFIX: (0,(5,5)),
                  'sink'+MODEL_SUFFIX: (0,(5,1)),
                  'opsink'+MODEL_SUFFIX: (0,(5,1)),
                  'dp'+MODEL_SUFFIX: (0,(1,1)),
                  'opdp'+MODEL_SUFFIX: (0,(1,1))
                  }

DEPTH_TO_MARKER = {1: '^', 2: 's', 3: 'p', 4: 'hexagon2'}

# NAMES_DICT = {'fnsformer': 'FNS', 
#               'spfnsformer': rf'FNS ($\mathbb{S}^{{d-1}}$)', 'spopfnsformer': rf'OPFNS ($\mathbb{S}^{{d-1}}$)',               
#               'rdfnsformer': rf'FNS ($\mathbb{R}^{{d}}$)', 'rdopfnsformer': rf'OPFNS ($\mathbb{R}^{{d}}$)',
#               'sinkformer': 'SINK',
#               'dpformer': 'DP',
#               'imdb': 'IMDb', 'rotten_tomatoes': 'Rotten Tomatoes',
#               'eval_loss': 'Loss', 'eval_accuracy': 'Accuracy', 'eval_f1_score': r'$F_1$ score'}              

DATASET_NAMES = ['rotten_tomatoes','imdb','emotion']     
MAX_LENS = [128, 512, 128]
MAX_LENS_DICT = {}
for ii, dataset_name in enumerate(DATASET_NAMES):
    MAX_LENS_DICT[dataset_name] = MAX_LENS[ii]