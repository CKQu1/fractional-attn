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

# paths = [".droot/length_500/attn_graph_results_1.2.npz", ".droot/length_500/attn_graph_results_2.0.npz", ".droot/length_500/attn_graph_results_dp.npz"]
# paths = [".droot/length_32/attn_graph_results_1.2.npz", ".droot/length_32/attn_graph_results_2.0.npz", ".droot/length_32/attn_graph_results_dp.npz"]

source_data_path = '../.source_data/fig5'
Path(source_data_path).parent.mkdir(parents=True, exist_ok=True)
L_d_path = '.droot/L-d-grid-v1mask-v7scaling-v2'

d = 8
shared_path = f'{L_d_path}/1L-hidden={d}-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv'
seed = 3
paths = [f"{shared_path}/oprdfnsformer-imdb-qqv-alpha=1.2-eps=1.0/model={seed}/attn_graph_results.npz",
         f"{shared_path}/oprdfnsformer-imdb-qqv-alpha=2.0-eps=1.0/model={seed}/attn_graph_results.npz",
         f"{shared_path}/opdpformer-imdb-qqv/model={seed}/attn_graph_results.npz",
        #  f"{shared_path}/opsinkformer-imdb-qqv/model={seed}/attn_graph_results.npz"
         ]

# PANEL B
##############################################################
#################### SHORTEST PATH MATRIX #################### 
##############################################################

seq_len = 500
shortest_path_lengths = []
shortest_path_lengths_check = []  # for checking
for i in range(len(paths)):
# for i in range(1):
    # path = njoin(parent_dir,paths[i])
    path = paths[i]
    # ax = axs[i]

    data_all = np.load(path)

    # ----- 1. compute from scratch (for checking) -----
    # attn_weights = np.squeeze(data_all["attention_weights"])[:seq_len, :seq_len]
    # edge_weights = 1/np.abs(attn_weights)
    # np.fill_diagonal(edge_weights, 0)
    # G = nx.from_numpy_array(edge_weights, create_using=nx.DiGraph)
    # shortest_path_length = np.full(attn_weights.shape, np.inf)
    # np.fill_diagonal(shortest_path_length, 0)
    # for source in tqdm(range(shortest_path_length.shape[0])):
    #     lengths, shortest_paths = nx.single_source_dijkstra(G, source) # Compute shortest paths using Dijkstra's algorithm
    #     for target in lengths:
    #         shortest_path_length[source, target] = len(shortest_paths[target]) - 1
    # shortest_path_lengths_check.append(shortest_path_length)
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

# titles = [r'$\alpha = 1.2$', r'$\alpha = 2.0$', 'DP', 'SINK']
titles = [r'$\alpha = 1.2$', r'$\alpha = 2.0$', 'DP']
gridspec = {'width_ratios': [1, 1, 1, 0.07]}
# gridspec = {'width_ratios': [1] * len(paths) + [0.07]}
figsize=(6,1.9)
# figsize=(5.5,1.74)
# figsize=(6.6,2.088)
# figsize=(4.5,1.425)
# figsize = (1.5*len(paths),1.9)
fig, axs = plt.subplots(1,len(paths)+1, gridspec_kw=gridspec, figsize=figsize)
source_j, source_i = np.indices(shortest_path_lengths[0].shape)
source_data_b = {'position_i': source_i.ravel(), 'position_j': source_j.ravel()}
for i in range(len(paths)):
    ax = axs[i]
    shortest_path_length = shortest_path_lengths[i]
    im = ax.imshow(shortest_path_length, vmin=0, vmax=14, cmap=cm.torch_r, origin='lower')
    source_label = titles[i].replace('$', '').replace(r'\alpha', 'alpha').replace(' ', '')
    source_data_b[source_label] = shortest_path_length.ravel()
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
        
cbar = fig.colorbar(im, cax=axs[len(paths)], fraction=0.000002)
cbar.ax.set_ylabel("Shortest path length")
cbar.ax.set_yticks([0, 7, 14])

plt.tight_layout()
from constants import FIGS_DIR
SAVE_DIR = njoin(FIGS_DIR, 'nlp-mechanisms')    
fig_file = 'shortest_path_matrix'
fig_file += '.pdf'
plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight', dpi=500)
pd.DataFrame(source_data_b).to_csv(f'{source_data_path}b.csv', index=False)
# plt.show()

# torch_r 
# rainforest_r
# freeze_r
# arctic_r
# amethyst_r

# PANEL C
##############################################################
#################### SHORTEST PATH EXAMPLE #################### 
##############################################################

import networkx as nx

# paths = [".droot/length_500/attn_graph_results_1.2.npz", ".droot/length_500/attn_graph_results_2.0.npz", ".droot/length_500/attn_graph_results_dp.npz"]
seed = 3
d = 8
shared_path = f'{L_d_path}/1L-hidden={d}-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv'
paths = [f"{shared_path}/oprdfnsformer-imdb-qqv-alpha=1.2-eps=1.0/model={seed}/attn_graph_results.npz",
         f"{shared_path}/oprdfnsformer-imdb-qqv-alpha=2.0-eps=1.0/model={seed}/attn_graph_results.npz",
         f"{shared_path}/opdpformer-imdb-qqv/model={seed}/attn_graph_results.npz"]

# Read from txt file
# path = ".droot/length_500/text_1.2.txt"
path = f"{shared_path}/oprdfnsformer-imdb-qqv-alpha=1.2-eps=1.0/model={seed}/text.txt"
with open(path, "r") as f:
    tokens = f.read().splitlines()

# Choose two tokens in the sequence 
# Studying these a little, we interestingly find the adjacent words "much" (pos 127) and "danger" (pos 128) actually require 13 steps to connect in DPformer
# Let's see how many steps it takes to connect them for fracformers...
seq_len = 500
token_idxs_to_plot = []
for i in range(3):
    path = paths[i]
    # ax = axs[i]
    attn_weights = np.squeeze(np.load(path)["attention_weights"])[:seq_len, :seq_len]
    edge_weights = 1/np.abs(attn_weights)
    np.fill_diagonal(edge_weights, 0)
    G = nx.from_numpy_array(edge_weights, create_using=nx.DiGraph)
    shortest_path_length = np.full(attn_weights.shape, np.inf)
    np.fill_diagonal(shortest_path_length, 0)
    lengths, shortest_paths = nx.single_source_dijkstra(G, 127) # Compute shortest paths using Dijkstra's algorithm
    print(f"Shortest path between {tokens[127]} and {tokens[128]}: {len(shortest_paths[128]) - 1}")
    token_idxs_to_plot.append(shortest_paths[128])
tokens_to_plot = [[tokens[i] for i in shortest_path] for shortest_path in token_idxs_to_plot]
print(tokens_to_plot)

# Find corresponding attention weights for plotting
edge_weights_to_plot = []
for i in range(3):
    path = paths[i]
    attn_weights = np.squeeze(np.load(path)["attention_weights"])[:seq_len, :seq_len]
    token_idxs = token_idxs_to_plot[i]
    edge_weights = []
    for i, j in zip(token_idxs[:-1], token_idxs[1:]):
        edge_weights.append( attn_weights[i, j] )
    edge_weights_to_plot.append(edge_weights)


from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

def add_query_to_key_arrows(ax, lc):
    """Mark each attention edge i -> j, where i is the query and j the key."""
    colors = lc.get_colors()
    for edge_idx, (query, key) in enumerate(lc.get_segments()):
        # Overlay only the arrowhead, leaving the tapered line width intact.
        arrow = FancyArrowPatch(
            query, key,
            arrowstyle='-|>', mutation_scale=8, linewidth=0,
            color=colors[edge_idx % len(colors)],
            shrinkA=0, shrinkB=3, zorder=lc.get_zorder() + 1,
        )
        ax.add_patch(arrow)

def taper_linecollection(lc, end_width, mid_widths, n_subsegments=100):
    """
    Modify a LineCollection so each line tapers:
    - linewidth = 2 at the ends (nodes)
    - linewidth = mid_widths[i] at the middle
    
    Parameters
    ----------
    lc : LineCollection
        The existing line collection (with N segments).
    mid_widths : list or array-like of length N
        Target linewidth at the midpoint of each line.
    n_subsegments : int
        How many subsegments to split each line into (for smooth taper).
    
    Returns
    -------
    new_lc : LineCollection
        A new LineCollection with tapered linewidths applied.
    """
    segments = lc.get_segments()
    new_segments = []
    new_linewidths = []
    
    for seg, mid_w in zip(segments, mid_widths):
        (x0, y0), (x1, y1) = seg
        # Interpolated points along the edge
        t = np.linspace(0, 1, n_subsegments+1)
        xs = np.linspace(x0, x1, n_subsegments+1)
        ys = np.linspace(y0, y1, n_subsegments+1)
        
        # Build subsegments
        pts = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

        # Linewidth profile: end_width at ends, mid_w at center
        lw_profile = end_width + (mid_w - end_width) * np.sin(np.pi * t[:-1])**2

        new_segments.extend(segs)
        new_linewidths.extend(lw_profile)
    
    new_lc = LineCollection(new_segments, linewidths=new_linewidths, colors=lc.get_colors(), antialiased=False, zorder=lc.get_zorder())
    return new_lc

def create_lc(X, Y, zorder=1):
    points = np.array([X, Y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return LineCollection(segments, linewidths=2, zorder=zorder)
# Rescale edge weights for visibility
# ewtp_rescaled = [[1/ew for ew in edge_weights] for edge_weights in edge_weights_to_plot]
# factor = 1 / np.max([np.max(ew) for ew in ewtp_rescaled])
# ewtp_rescaled = [[ew * factor for ew in edge_weights] for edge_weights in ewtp_rescaled]
factor = 1 / np.min([np.min(ew) for ew in edge_weights_to_plot])
# factor_scale = 0.25
factor_scale = 0.5
ewtp_rescaled = [[ew * factor * factor_scale for ew in edge_weights] for edge_weights in edge_weights_to_plot]
# Plot 
words = [tokens_to_plot[1][1]] + tokens_to_plot[2]
print(words)
fig, ax = plt.subplots(figsize=(2,2))


# Draw words
# theta = np.linspace(0, 360, 15, endpoint=False)

total_unique_tokens = 0
for token_ls in tokens_to_plot:
    total_unique_tokens += len(token_ls)
total_unique_tokens -= 6
theta = np.linspace(0, 360, total_unique_tokens, endpoint=False)


x = np.cos(np.radians(theta) + np.pi/2)
y = np.sin(np.radians(theta) + np.pi/2)
text_radius = 1.45
ax.plot(x, y, 'k.', zorder=5)
source_data_c = {
    'node_x': pd.Series(x), 'node_y': pd.Series(y),
    'text_x': pd.Series(-x[:len(words)] * text_radius),
    'text_y': pd.Series(y[:len(words)] * text_radius),
}
source_words = []
for i in range(len(words)):
    word = words[i]
    if not word[0].isalpha():
        word = word[1:]
    source_words.append(word)
    ax.annotate(word, (-x[i]*text_radius, y[i]*text_radius), size=6, ha='center', va='center')
source_data_c['token'] = pd.Series(source_words)
# attention_weights[i, j] is query i attending to key j.
# Coordinates follow the directed path; weights and widths describe outgoing edges.
# Draw 1.2 attention
X = np.array([x[1], x[-1]])
Y = np.array([y[1], y[-1]])
lc = create_lc(X,Y, zorder=3)
lc.set_color("#2E63A6")
add_query_to_key_arrows(ax, lc)
# lc.set_alpha(ewtp_rescaled[0])
# lc.set_linewidths(ewtp_rescaled[0])
lc = taper_linecollection(lc, 1, ewtp_rescaled[0], n_subsegments=500)
ax.add_collection(lc)
source_data_c['alpha=1.2_x'] = pd.Series(X)
source_data_c['alpha=1.2_y'] = pd.Series(Y)
source_data_c['alpha=1.2_edge_weight'] = pd.Series(edge_weights_to_plot[0][:len(X)-1])
source_data_c['alpha=1.2_edge_mid_width'] = pd.Series(ewtp_rescaled[0][:len(X)-1])
# ax.plot([x[1], x[-1]], [y[1], y[-1]], c="#2E63A6", lw=2, zorder=-3)
# Draw 2.0 attention
# ax.plot([x[1], x[0], x[-1]], [y[1], y[0], y[-1]], c="#A4292F", lw=2, zorder=-3)
X = np.array([x[1], x[0], x[-1]])
Y = np.array([y[1], y[0], y[-1]])
lc = create_lc(X,Y)
lc.set_color("#A4292F")
add_query_to_key_arrows(ax, lc)
# lc.set_alpha(ewtp_rescaled[1])
# lc.set_linewidths(ewtp_rescaled[1])
lc = taper_linecollection(lc, 1, ewtp_rescaled[1], n_subsegments=500)
ax.add_collection(lc)
source_data_c['alpha=2.0_x'] = pd.Series(X)
source_data_c['alpha=2.0_y'] = pd.Series(Y)
source_data_c['alpha=2.0_edge_weight'] = pd.Series(edge_weights_to_plot[1][:len(X)-1])
source_data_c['alpha=2.0_edge_mid_width'] = pd.Series(ewtp_rescaled[1][:len(X)-1])
# Draw DP attention
# ax.plot(x[1:], y[1:], c="#636363", lw=2, zorder=-3)
X = np.array(x[1:])
Y = np.array(y[1:])
lc = create_lc(X,Y)
lc.set_color("#636363")
add_query_to_key_arrows(ax, lc)
# lc.set_alpha(ewtp_rescaled[2])
# lc.set_linewidths(ewtp_rescaled[2])
lc = taper_linecollection(lc, 1, ewtp_rescaled[2], n_subsegments=500)
ax.add_collection(lc)
source_data_c['DP_x'] = pd.Series(X)
source_data_c['DP_y'] = pd.Series(Y)
source_data_c['DP_edge_weight'] = pd.Series(edge_weights_to_plot[2][:len(X)-1])
source_data_c['DP_edge_mid_width'] = pd.Series(ewtp_rescaled[2][:len(X)-1])
plt.axis("equal")
plt.xlim([-1.6, 1.6])
plt.ylim([-1.6, 1.6])
plt.axis("off")
plt.tight_layout()
from constants import FIGS_DIR
SAVE_DIR = njoin(FIGS_DIR, 'nlp-mechanisms')    
fig_file = f'shortest_path_example-seed={seed}'
fig_file += '.pdf'
plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')
pd.DataFrame(source_data_c).to_csv(f'{source_data_path}c.csv', index_label='index')
# plt.show()


# PANEL A
##############################################################
#################### MEAN SPECTRAL GAP #################### 
##############################################################

# only for testing, does not affect results after this cell

# paths = [".droot/L-d-grid/1L-hidden=8-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv/oprdfnsformer-imdb-qqv-alpha=1.2-eps=1.0/model=0/andrew_results_2.npz",
#          ".droot/L-d-grid/1L-hidden=8-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv/oprdfnsformer-imdb-qqv-alpha=2.0-eps=1.0/model=0/andrew_results_2.npz",
#         ".droot/L-d-grid/1L-hidden=8-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv/opdpformer-imdb-qqv/model=0/andrew_results_2.npz"]

d = 8
shared_path = f'{L_d_path}/1L-hidden={d}-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv'
paths = [f"{shared_path}/oprdfnsformer-imdb-qqv-alpha=1.2-eps=1.0/model=0/andrew_results_2.npz",
         f"{shared_path}/oprdfnsformer-imdb-qqv-alpha=2.0-eps=1.0/model=0/andrew_results_2.npz",
         f"{shared_path}/opdpformer-imdb-qqv/model=0/andrew_results_2.npz"
         ]


for i in range(3):
    path = paths[i]
    mean_spectral_gaps = np.load(paths[i])["mean_spectral_gaps"]


# for checking the spectrum

# seed = 4
seed = 3
# model_root = '.droot/L-d-grid-v2/1L-hidden=8-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv'
model_root = f'{L_d_path}/1L-hidden=8-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv'
paths = [f"{model_root}/oprdfnsformer-imdb-qqv-alpha=1.2-eps=1.0/model=0/spectrum_file.npz",
         f"{model_root}/oprdfnsformer-imdb-qqv-alpha=2.0-eps=1.0/model=0/spectrum_file.npz",
         f"{model_root}/opdpformer-imdb-qqv/model=0/spectrum_file.npz"]

eidx = 10
for i in range(len(paths)):
    path = paths[i]
    spectrum = np.load(paths[i])["spectrum"]

    print(spectrum.shape)
    i_max = np.where(spectrum[eidx,:] == spectrum[eidx,:].max())[0].item()
    print(i_max)
    print(spectrum[eidx,:].max())
    print(spectrum[eidx,:i_max].max())
    print('\n')
#     print(spectrum[0,i_max-10:i_max+1])
#     print(spectrum[0,i_max])
#     print(spectrum[0,i_max-1])
#     print(spectrum[0,i_max:])


# getting the model performances
# def get_all_test_results():
is_rescale_dist=True
fns_manifold='rd'
# models_root = ".droot/L-d-grid-v2/"
models_root = L_d_path
qk_shares = [True]
selected_alphas='1.0,1.2,1.4,1.6,1.8,2'
metric='val_acc'
selected_dataset='imdb'
is_op=True

fns_type = fns_manifold + 'fns' + MODEL_SUFFIX 
other_model_type = 'dpformer'
if is_op:
    fns_type = 'op' + fns_type
    other_model_type = 'op' + other_model_type
model_types_to_plot = [fns_type, other_model_type]

is_op, is_rescale_dist = str2bool(is_op), str2bool(is_rescale_dist)
qk_shares = str2ls(qk_shares)        
selected_alphas = [float(selected_alpha) for selected_alpha in str2ls(selected_alphas)]
# selected_alphas = [1.2, 2.0]  # delete
eps = 1

# Regular expression pattern
pattern = r"\d+L-hidden=\d+-max_len=512"
if is_rescale_dist:            
    pattern += "-rescaled"

# Extract matching subfolders
layer_dirs_dict = {}
layers, emb_ds = [], []
for layer_dir in os.listdir(models_root):
    is_match = re.fullmatch(pattern, layer_dir)
    if is_match:
        #layer, emb_d = int(is_match.group(1)), int(is_match.group(2))
        layer = int(layer_dir.split('L')[0])          
        #emb_d = int(layer_dir.split('-')[1].split('=')[1])
        emb_d = int(layer_dir.split('-')[1].split('=')[1])  
        if isdir(njoin(models_root, layer_dir)):
            layer_dirs_dict[f'{layer}-{emb_d}'] = njoin(models_root, layer_dir)
        layers.append(layer)
        emb_ds.append(emb_d)
layers = np.array(sorted(list(set(layers)))); layers = layers[layers < 2]
emb_ds = np.array(sorted(list(set(emb_ds)))); emb_ds = emb_ds[emb_ds < 65]

#nrows, ncols = len(qk_shares), len(selected_alphas)
nrows, ncols = len(qk_shares), len(layers)

# (model_types, qk_share, L, d_model)
N_model_types = len(selected_alphas) + 1
metric_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds), 5])
gap_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds), 5])

# average_metric_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
# std_metric_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
# average_metric_matrix[:] = np.nan
# std_metric_matrix[:] = np.nan
model_types_plotted = []
for (qk_ii,qk_share),(layer_idx,layer),(emb_d_idx,emb_d) in\
        product(enumerate(qk_shares),enumerate(layers),enumerate(emb_ds)):

    qk_share_dirname = 'config_qqv' if qk_share else 'config_qkv'
    print(f'qk_share = {qk_share}, layer = {layer}, emb_d = {emb_d}')    
    # directories matching the above setting in the triple for loop
    if f'{layer}-{emb_d}' in layer_dirs_dict.keys():
        if qk_share_dirname in os.listdir(layer_dirs_dict[f'{layer}-{emb_d}']):
            setting_dir = njoin(layer_dirs_dict[f'{layer}-{emb_d}'], qk_share_dirname)
        else:
            continue
    else:
        continue
    # for _ in range(2):
    #     setting_dir = njoin(setting_dir, os.listdir(setting_dir)[0])
    setting_dir = njoin(setting_dir, 'imdb')
    setting_dir = njoin(setting_dir, os.listdir(setting_dir)[0])
    DCT_ALL = collect_model_dirs(setting_dir, suffix=MODEL_SUFFIX)
    model_df = DCT_ALL[fns_type].dropna(subset='alpha')
    model_df.reset_index(drop=True, inplace=True)

    for model_type in model_types_to_plot:
        if model_type in DCT_ALL.keys():
            df_model = DCT_ALL[model_type]
        else:
            continue
        condition0 = (df_model['ensembles']>0)&(df_model['qk_share']==qk_share)&\
                     (df_model['is_op']==is_op)&(df_model['model_dir'].str.contains(selected_dataset))&\
                     (df_model['model_dir'].str.contains(f'{model_type}-'))
        matching_df = df_model[condition0]

        if model_type not in model_types_plotted:
            model_types_plotted.append(model_type)
        lstyle_model = LINESTYLE_DICT[model_type]            
        for alpha_idx, alpha in enumerate(selected_alphas):  
            # if is fns type
            is_fns = 'fns' in model_type
            alpha = alpha if is_fns else None
            matching_df.reset_index(drop=True, inplace=True)

            # -------------------- SINK, DP -------------------- 
            model_info = matching_df 
            # -------------------- FNS --------------------
            if is_fns:
                condition = (matching_df['alpha']==alpha) & (matching_df['bandwidth']==eps)
                model_info = model_info[condition]
            else:
                alpha_idx = len(selected_alphas)

            # load model performance
            if model_info.shape[0] > 0:
                seeds, qk_share = (model_info[k].item() for k in ('seeds', 'qk_share'))                
                epochs, run_perf_all = load_seed_runs(model_info['model_dir'].item(), seeds, metric)   
            else:
                continue
            if run_perf_all is not None:
                metric_matrix[qk_ii, alpha_idx, layer_idx, emb_d_idx, :] = run_perf_all.loc[run_perf_all.index[-1]:, metric]

            # load mean spectral gap

            for seed in seeds:
                gap_matrix[qk_ii, alpha_idx, layer_idx, emb_d_idx, seed] =\
                    np.load(njoin(model_info['model_dir'].item(), 
                                  f'model={seed}', 'andrew_results_2.npz'))["mean_spectral_gaps"].item() 

            if not is_fns:
                break  # only do once if model is NOT FNS type   
    # return metric_matrix, gap_matrix

# metric_matrix = get_all_test_results()
# metric_matrix, gap_matrix = get_all_test_results()   


# update version
from matplotlib.colors import to_rgb

# data = np.load(".droot/all_spectral_gap.npz")
# spectral_gap = data["spectral_gap"]

# X_lens = data["X_lens"]
# alphas = data["alphas"]
# dimensions = data["dimensions"]

# metric_matrix = get_all_test_results()
# metric_matrix, spectral_gap = get_all_test_results()
spectral_gap = gap_matrix[0,:,0,:,:]  # [model_types, emb_ds, seeds]
# alphas = [1.0, 1.2, 1.4, 1.6, 1.8, 2, None]
alphas = selected_alphas + [None]
    
colors = ["#636363", "#469C76", "#2E63A6", "#C17DA5", "#C66526", "#EEE461", "#A4292F"]
# colors = ["#636363", "#2E63A6", "#A4292F"]
# marker_alphas = [1, 0.8, 0.6, 0.4]
marker_alphas = [1] * 4
marker_sizes = [8,6,4,2]
fig, ax = plt.subplots(figsize=(1.5,1.5))
source_data_a = {'hidden_size': emb_ds[:len(marker_sizes)]}
for model_idx in range(len(alphas)):
    if model_idx == 0: # DPformer
        idx = -1
    else:
        idx = model_idx - 1
    test_accs = np.mean(metric_matrix[0,idx,0,:,:], axis=-1)
    # ----- [nrows, N_model_types, len(layers), len(emb_ds), 5] -----
    # gap = np.mean(spectral_gap[model_idx, :, :], axis=-1)
    gap = np.mean(spectral_gap[idx, :, :], axis=-1)
    color = colors[model_idx]
    r, g, b = to_rgb(color)
    color = np.array([(r, g, b, alpha) for alpha in marker_alphas])
    # xerr = np.std(spectral_gap[model_idx, :, :], axis=-1)
    xerr = np.std(spectral_gap[idx, :, :], axis=-1, ddof=1)
    yerr = np.std(metric_matrix[0,idx,0,:,:], axis=-1, ddof=1)
    # ax.scatter(gap, test_accs, clip_on=False, label='DP' if model_idx==0 else rf'$\alpha={alphas[idx]:.1f}$', color=color, s=marker_sizes)
    if model_idx == 0:
        marker_style = 'D'
        size_factor = 0.5
    elif model_idx == 6:
        marker_style = 's'
        size_factor = 0.5
    else:
        marker_style = '.'
        size_factor = 1
    source_label = 'DP' if model_idx == 0 else f'alpha={alphas[idx]:.1f}'
    source_data_a[f'{source_label}_spectral_gap'] = gap[:len(marker_sizes)]
    source_data_a[f'{source_label}_accuracy'] = test_accs[:len(marker_sizes)]
    source_data_a[f'{source_label}_xerr'] = xerr[:len(marker_sizes)]
    source_data_a[f'{source_label}_yerr'] = yerr[:len(marker_sizes)]
    for plot_idx, (s, g, ta, xe, ye) in enumerate(zip(marker_sizes, gap, test_accs, xerr, yerr)):
        if plot_idx == 0:
            ax.errorbar(g, ta, xerr=xe, yerr=ye, fmt=marker_style, c=color[0], markersize=s*size_factor, label='DP' if model_idx==0 else rf'$\alpha={alphas[idx]:.1f}$', elinewidth=1)
        else:
            ax.errorbar(g, ta, xerr=xe, yerr=ye, fmt=marker_style, c=color[0], markersize=s*size_factor, elinewidth=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(top=86)
# ax.set_yticks([78, 82, 86])
ax.set_yticks(list(range(70, 87, 4)))
# ax.set_yticks(list(range(70, 91, 6)))
ax.set_xlim(right=1)
# ax.set_xlim(right=0.18)
# ax.set_xticks([0, 0.06, 0.12, 0.18])
ax.set_xlabel('Spectral gap')
ax.set_ylabel('Testing accuracy (%)')
ax.legend(frameon=False, bbox_to_anchor=(1, 1.05))
from constants import FIGS_DIR
SAVE_DIR = njoin(FIGS_DIR, 'nlp-mechanisms')  
if not isdir(SAVE_DIR): makedirs(SAVE_DIR)  
plt.savefig(njoin(SAVE_DIR, "spectral_gap_vs_acc.pdf"), bbox_inches='tight')
pd.DataFrame(source_data_a).to_csv(f'{source_data_path}a.csv', index=False)
# plt.show()


# PANEL D
##############################################################
#################### MEAN SHORTEST PATH #################### 
##############################################################

# paths = [".droot/attn_graph_results_1.2.npz", ".droot/attn_graph_results_2.0.npz", ".droot/attn_graph_results_dp.npz"]

# seed = 4
seed = 3
d = 8
shared_path = f'{L_d_path}/1L-hidden={d}-max_len=512-rescaled/config_qqv/imdb/layers=1-heads=1-qqv'
paths = [f"{shared_path}/oprdfnsformer-imdb-qqv-alpha=1.2-eps=1.0/model={seed}/attn_graph_results.npz",
         f"{shared_path}/oprdfnsformer-imdb-qqv-alpha=2.0-eps=1.0/model={seed}/attn_graph_results.npz",
         f"{shared_path}/opdpformer-imdb-qqv/model={seed}/attn_graph_results.npz"]
seq_lens = np.arange(50, 501, 50)

plt.close('all')
fig, ax = plt.subplots()
# fig.set_size_inches(1.5,1.5)
fig.set_size_inches(4.3,1.5)
colors = ["#2E63A6", "#A4292F", "#636363"]
source_data_d = {'seq_len': seq_lens}
for i, path in enumerate(paths):
    data = np.load(path)
    shortest_path_lengths = data["shortest_path_lengths"]
    shortest_path_lengths[shortest_path_lengths == 0] = np.nan
    mean_shortest_path_lengths = np.nanmean(shortest_path_lengths, axis=-1)
    std_shortest_path_lengths = np.nanstd(shortest_path_lengths, axis=-1, ddof=1)
    source_label = 'DP' if 'dp' in path else ('alpha=1.2' if '1.2' in path else 'alpha=2.0')
    source_data_d[f'{source_label}_mean'] = mean_shortest_path_lengths
    source_data_d[f'{source_label}_yerr'] = std_shortest_path_lengths / 5
    ax.errorbar(seq_lens, mean_shortest_path_lengths, yerr=std_shortest_path_lengths/5, label='DP' if 'dp' in path else (r'$\alpha=1.2$' if '1.2' in path else r'$\alpha=2.0$'), elinewidth=1, color=colors[i], marker='o', markersize=2)
ax.legend(frameon=False, bbox_to_anchor=(1, 1))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Sequence length')
ax.set_ylabel('Shortest path length')
ax.set_yticks([1,2,3,4,5])
ax.set_xticks([100, 200, 300, 400, 500])
from constants import FIGS_DIR
SAVE_DIR = njoin(FIGS_DIR, 'nlp-mechanisms')  
if not isdir(SAVE_DIR): makedirs(SAVE_DIR)   
plt.savefig(njoin(SAVE_DIR, f"shortest_path_length_vs_seq_len-seed={seed}.pdf"), bbox_inches='tight')
pd.DataFrame(source_data_d).to_csv(f'{source_data_path}d.csv', index=False)
# plt.show()