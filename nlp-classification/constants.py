import os
from mutils import njoin

PROJECTS = ['phys_DL','PDLAI','dnn_maths','dyson','vortex_dl','frac_attn']
#DROOT = njoin(os.path.abspath(os.getcwd()), 'droot')
DROOT = njoin('/project/frac_attn/fractional-attn/nlp-classification', '.droot')
FIGS_DIR = njoin(DROOT, 'figs_dir')

BPATH = njoin('/project')  # path for binding to singularity container
SPATH = njoin('/project/frac_attn/built_containers/FaContainer_v3.sif')  # singularity container path

MODEL_NAMES = ['fnsformer', 
               'spfnsformer','spopfnsformer',
               'rdfnsformer', 'rdopfnsformer',
               'sinkformer',
               'dpformer', 
               'l2former']  # model str names
               
NAMES_DICT = {'fnsformer': 'FNS', 
              'spfnsformer': r'FNS (\mathbb{S}^{{d-1}})', 'spopfnsformer': r'OPFNS (\mathbb{S}^{{d-1}})',               
              'rdfnsformer': r'FNS (\mathbb{R}^{{d}})', 'rdopfnsformer': r'OPFNS (\mathbb{R}^{{d}})',
              'sinkformer': 'SINK',
              'dpformer': 'DP',
              'imdb': 'IMDb', 'rotten_tomatoes': 'Rotten Tomatoes',
              'eval_loss': 'Loss', 'eval_accuracy': 'Accuracy', 'eval_f1_score': r'$F_1$ score'}