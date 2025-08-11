import matplotlib as mpl
import os
from mutils import njoin
from matplotlib.cm import get_cmap

# ----- GENERAL -----
RT = os.path.abspath(os.getcwd())
DROOT = njoin(RT, '.droot')
FIGS_DIR = njoin(DROOT, 'figs_dir')
SCRIPT_DIR = njoin(DROOT, 'submitted_scripts')

CLUSTER = 'ARTEMIS' if 'project' in DROOT else 'PHYSICS' if 'taiji1' in DROOT else 'FUDAN_BRAIN'
# -------------------

# ----- ARTEMIS -----
PROJECTS = ['phys_DL','PDLAI','dnn_maths','dyson','vortex_dl','frac_attn', 'ddl']
BPATH = njoin('/project')  # path for binding to singularity container
#SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v5.sif')  # singularity container path
SPATH = njoin('/project/frac_attn/built_containers/pydl.img')
# -------------------

# ----- PHYSICS -----
PHYSICS_SOURCE = '/usr/physics/python/Anaconda3-2022.10/etc/profile.d/conda.sh'
#PHYSICS_CONDA = 'frac_attn' if 'chqu7424' in RT else '~/conda'
PHYSICS_CONDA = '/taiji1/chqu7424/myenvs/pydl' if 'chqu7424' in RT else '~/conda'
# -------------------

# ----- FUDAN-BRAIN -----
FUDAN_CONDA = 'frac_attn'
# -------------------

# ----- MODELS -----

MODEL_PREFIXES = ['fns', 'opfns', 'spfns', 'spopfns', 'rdfns', 'oprdfns', 'sink', 'opsink', 'dp', 'opdp']
MODEL_SUFFIX = 'former'

MODEL_NAMES = []
NAMES_DICT = {}
for MODEL_PREFIX in MODEL_PREFIXES:
    MODEL_NAMES.append(MODEL_PREFIX + MODEL_SUFFIX)

    if 'fns' in MODEL_PREFIX.lower():
        if 'rd' in MODEL_PREFIX.lower():
            NAMES_DICT[MODEL_PREFIX + MODEL_SUFFIX] = r'FRAC $(\mathbb{{R}}^d)$'
        else:                    
            NAMES_DICT[MODEL_PREFIX + MODEL_SUFFIX] = r'FRAC $(\mathcal{{S}}^{{d-1}})$'
    else:
        NAMES_DICT[MODEL_PREFIX + MODEL_SUFFIX] = MODEL_PREFIX.upper()

# ----- TRAINING -----

DATASET_NAMES = ['imdb-classification', 'lra-cifar-classification',
                 'listops-classification', 'aan-classification',
                 'pathfinder-classification', 'pathx-classification']     

DATASET_EPOCHS = {'imdb-classification':20, 'lra-cifar-classification':200,
                  'listops-classification':20, 'aan-classification':20,
                  'pathfinder-classification':200, 'pathx-classification':200}                

DATASET_STEPS = {'imdb-classification':100000, 'lra-cifar-classification':100000,
                  'listops-classification':500000, 'aan-classification':500000,
                  'pathfinder-classification':500000, 'pathx-classification':500000}                                