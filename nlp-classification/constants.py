import matplotlib as mpl
import os
from mutils import njoin
from matplotlib.cm import get_cmap

PROJECTS = ['phys_DL','PDLAI','dnn_maths','dyson','vortex_dl','frac_attn']
#DROOT = njoin(os.path.abspath(os.getcwd()), 'droot')
DROOT = njoin('/project/frac_attn/fractional-attn/nlp-classification', '.droot')
FIGS_DIR = njoin(DROOT, 'figs_dir')

BPATH = njoin('/project')  # path for binding to singularity container
SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v4.sif')  # singularity container path

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