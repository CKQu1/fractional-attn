import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import math
import numpy as np
import pandas as pd
import re

from ast import literal_eval
from itertools import product
from matplotlib.transforms import ScaledTranslation
from matplotlib.ticker import NullFormatter
from os import makedirs
from os.path import isdir, isfile
from pathlib import Path
from string import ascii_lowercase
from time import time
from tqdm import tqdm
from constants import *
from UTILS.mutils import njoin, str2bool, str2ls, create_model_dir, convert_train_history
from UTILS.mutils import collect_model_dirs, find_subdirs, load_model_files
from UTILS.figure_utils import matrixify_axs, label_axs
from plot_results import *

import networkx as nx
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cm

# paths = [".droot/L-d-grid/1L-hidden=8-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv/oprdfnsformer-imdb-qqv-alpha=1.2-eps=1.0/model=0/andrew_results_1.npz",
#          ".droot/L-d-grid/1L-hidden=8-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv/oprdfnsformer-imdb-qqv-alpha=2.0-eps=1.0/model=0/andrew_results_1.npz",
#         ".droot/L-d-grid/1L-hidden=8-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv/opdpformer-imdb-qqv/model=0/andrew_results_1.npz"]
# paths = [".droot/length_500/attn_graph_results_1.2.npz", ".droot/length_500/attn_graph_results_2.0.npz", ".droot/length_500/attn_graph_results_dp.npz"]
# paths = [".droot/length_32/attn_graph_results_1.2.npz", ".droot/length_32/attn_graph_results_2.0.npz", ".droot/length_32/attn_graph_results_dp.npz"]

# parent_dir = 'U:/scratch/uu69/cq5024/projects/fractional-attn/nlp-tutorial'
shared_path = '.droot/L-d-grid-v2/1L-hidden=8-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv'
seed = 0
paths = [f"{shared_path}/oprdfnsformer-imdb-qqv-alpha=1.2-eps=1.0/model={seed}/attn_graph_results.npz",
         f"{shared_path}/oprdfnsformer-imdb-qqv-alpha=2.0-eps=1.0/model={seed}/attn_graph_results.npz",
         f"{shared_path}/opdpformer-imdb-qqv/model={seed}/attn_graph_results.npz"]

seq_len = 500
shortest_path_lengths = []
shortest_path_lengths_check = []
for i in range(3):
# for i in range(1):
    # path = njoin(parent_dir,paths[i])
    path = paths[i]
    # ax = axs[i]

    data_all = np.load(path)

    # ----- 1. compute from scratch -----
    attn_weights = np.squeeze(data_all["attention_weights"])[:seq_len, :seq_len]
    edge_weights = 1/np.abs(attn_weights)
    np.fill_diagonal(edge_weights, 0)
    G = nx.from_numpy_array(edge_weights, create_using=nx.DiGraph)
    shortest_path_length = np.full(attn_weights.shape, np.inf)
    np.fill_diagonal(shortest_path_length, 0)
    for source in tqdm(range(shortest_path_length.shape[0])):
        lengths, shortest_paths = nx.single_source_dijkstra(G, source) # Compute shortest paths using Dijkstra's algorithm
        for target in lengths:
            shortest_path_length[source, target] = len(shortest_paths[target]) - 1
    shortest_path_lengths_check.append(shortest_path_length)
    # ------------------------------------

    # ----- 2. load from data_all -----
    shortest_path_length = data_all["shortest_path_lengths"][-1].reshape(seq_len,seq_len)  # last one
    # ------------------------------------
    
    shortest_path_lengths.append(shortest_path_length)

from scipy.cluster.hierarchy import linkage, leaves_list
def agglomerative_reorder(distance_matrix, method="average"):
    """
    Perform agglomerative clustering on a distance matrix and 
    return a reordered matrix suitable for visualization.
    
    Parameters
    ----------
    distance_matrix : ndarray (n x n)
        Symmetric distance matrix.
    method : str
        Linkage method ('single', 'complete', 'average', 'ward', etc.).
        
    Returns
    -------
    reordered_matrix : ndarray (n x n)
        Distance matrix reordered according to clustering.
    order : ndarray (n,)
        The order of indices after clustering.
    """
    # Ensure distance matrix is square
    n, m = distance_matrix.shape
    assert n == m, "Distance matrix must be square"
    
    # Convert to condensed form (upper triangular as 1D vector)
    condensed = distance_matrix[np.triu_indices(n, k=1)]
    
    # Perform hierarchical clustering
    Z = linkage(condensed, method=method)
    
    # Get the order of leaves after clustering
    order = leaves_list(Z)
    
    # Reorder the distance matrix
    reordered_matrix = distance_matrix[np.ix_(order, order)]
    
    return reordered_matrix, order

print('Plot figures.')
titles = [r'$\alpha = 1.2$', r'$\alpha = 2.0$', 'DP']
gridspec = {'width_ratios': [1, 1, 1, 0.07]}
fig, axs = plt.subplots(1,4, gridspec_kw=gridspec, figsize=(6,1.9))
for i in range(3):
    ax = axs[i]
    shortest_path_length = shortest_path_lengths[i]
    im = ax.imshow(shortest_path_length, vmin=0, vmax=14, cmap=cm.torch_r, origin='lower')
    print("Max shortest path:", np.nanmax(shortest_path_length))
    # reordered_matrix, order = agglomerative_reorder(shortest_path_length, method="complete")
    # im = ax.imshow(reordered_matrix, vmin=0, vmax=14, cmap=cm.torch_r)
    print("Mean shortest path:", shortest_path_length.mean())
    ax.set_aspect(1)
    ax.set_title(titles[i])
    ax.set_xlabel(r'Position index $i$')
    ax.set_xticks([0,250,500])
    ax.set_yticks([0,250,500])
    if i == 0:
        ax.set_ylabel(r'Position index $j$')
        
cbar = fig.colorbar(im, cax=axs[3], fraction=0.000002)
cbar.ax.set_ylabel("Shortest path length")
cbar.ax.set_yticks([0, 7, 14])

plt.tight_layout()
from constants import FIGS_DIR
SAVE_DIR = njoin(FIGS_DIR, 'nlp-mechanisms')   
if not isdir(SAVE_DIR): makedirs(SAVE_DIR)  
fig_file = 'shortest_path_matrix'
fig_file += '.pdf'
plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight', dpi=500)
plt.show()

# torch_r 
# rainforest_r
# freeze_r
# arctic_r
# amethyst_r