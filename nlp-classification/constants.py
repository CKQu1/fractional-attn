import os
from mutils import njoin

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

# NAMES_DICT = {'fnsformer': 'FNS', 
#               'spfnsformer': rf'FNS ($\mathbb{S}^{{d-1}}$)', 'spopfnsformer': rf'OPFNS ($\mathbb{S}^{{d-1}}$)',               
#               'rdfnsformer': rf'FNS ($\mathbb{R}^{{d}}$)', 'rdopfnsformer': rf'OPFNS ($\mathbb{R}^{{d}}$)',
#               'sinkformer': 'SINK',
#               'dpformer': 'DP',
#               'imdb': 'IMDb', 'rotten_tomatoes': 'Rotten Tomatoes',
#               'eval_loss': 'Loss', 'eval_accuracy': 'Accuracy', 'eval_f1_score': r'$F_1$ score'}              