import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import math
import numpy as np
import os
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

matplotlib.use("Agg")

# ---------- Global plot settings ----------
# font_type = {'family' : 'sans-serif'}
# plt.rc('font', **font_type)
# plt.rc('legend',fontsize=7)
linestyles = ['-', '--', '-.', ':']
#linestyles = ['-', '--', ':']
markers = ['s', 'D', 'd', 'v', '^', 'o', '.']
markersize = '3'
colors = list(mcl.TABLEAU_COLORS.keys())
COLORS_ALPHA = ["#636363", "#469C76", "#2E63A6", "#C17DA5", "#C66526", "#EEE461", "#A4292F"]
# ------------------------------------------

MARKERSIZE = 4
#BIGGER_SIZE = 10
BIGGER_SIZE = 8
LEGEND_SIZE = 7
TRANSP = 1  # transparency (corresponding to alpha in plot)
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# -------------------- FUNCTIONS --------------------
# return median, 25/75 percentile
def get_metric_curves(run_perf_all,type='median'):
    if type == 'mean':
        metric_mean = run_perf_all.mean(axis=1)
        metric_std = run_perf_all.std(axis=1, ddof=1)
        metric_l = metric_mean - metric_std
        metric_u = metric_mean + metric_std
    elif type == 'median':
        metric_mean = run_perf_all.median(axis=1)
        metric_l = run_perf_all.quantile(0.25, axis=1)
        metric_u = run_perf_all.quantile(0.75, axis=1)

    return [metric_l, metric_mean, metric_u]

# aggregate all runs
def load_seed_runs(model_dir, seeds, metric):
    runs = []
    for seed in seeds:
        seed_path = njoin(model_dir, f'model={seed}')
        fpath = njoin(seed_path, 'run_performance.csv')
        if not isfile(fpath):
            fpath = njoin(seed_path, '_run_performance.csv')
            if not isfile(fpath):
                continue
            # continue
        run = pd.read_csv(fpath)
        if int(run.loc[0,'iter']) > 0:
            epochs = run['iter'].astype(int) // int(run.loc[0,'iter'])
        else:
            epochs = run['iter'].astype(int) // int(run.loc[1,'iter'])
        if 'acc' in metric and run.loc[run.index[-1], metric] <= 1:
            run[metric] *= 100
        runs.append(run[metric])
    if len(runs)==0:
        return (None, None)
    else:
        return epochs, pd.concat(runs, axis=1)

# final epoch stats
def final_epoch_stats(run_perf_all):
    metric_min = run_perf_all.tail(1).min(1).item()
    metric_max = run_perf_all.tail(1).max(1).item()
    metric_mid = (metric_min + metric_max) / 2

    metric_median = run_perf_all.tail(1).median(1).item()
    metric_mean = run_perf_all.tail(1).mean(1).item()
    metric_std = run_perf_all.tail(1).std(1).item()    
    return [metric_min, metric_max, metric_mid, metric_median, metric_mean, metric_std]
# --------------------------------------------------


# Plots average of metrics over ensembles (assumption 1 and 2 possibilities for full-sized models)
"""
python -i plot_results.py phase_ensembles .droot/L-d-grid/1L-hidden=8-max_len=512-rescaled/
python -i plot_results.py phase_ensembles frac_attn/fractional-attn/nlp-tutorial/droot/6L-hidden=256-max_len=None-rescaled (Figure 2)
"""
def phase_ensembles(models_root, selected_dataset='imdb', source_data_path='../.source_data/fig7',
                    qk_share=False, is_ops='False,True',
                    fns_manifold='rd', selected_alphas='1.2,2',
                    metrics='val_acc,val_loss',   #metric_type='mean',  # 'median'
                    cbar_separate=False, display=False):
    """Save each panel's curves and shaded bounds to <source_data_path><panel>.csv."""
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    global qk_shares, summary_stats, run_perf_all, metric_curves

    assert fns_manifold in ['sp', 'rd', 'v2_rd'], f'{fns_manifold} does not exist!'
    qk_share, cbar_separate, display = map(str2bool, (qk_share, cbar_separate, display))
    metrics, is_ops = str2ls(metrics), str2ls(is_ops)
    is_ops = [str2bool(is_op) for is_op in is_ops]

    # collect subdirs containing the model directories
    model_root_dirs = models_roots = find_subdirs(models_root, MODEL_SUFFIX)
    print(model_root_dirs)                  

    # all trained model types
    model_types = []   
    DCT_ALL = {} 
    for model_root_dir in model_root_dirs:
        DCT_cur = collect_model_dirs(model_root_dir, suffix=MODEL_SUFFIX)
        for model_type, df_model_cur in DCT_cur.items():
            df_clean = df_model_cur.dropna(subset='alpha') if 'alpha' in df_model_cur.columns else df_model_cur
            if model_type not in DCT_ALL:
                model_types.append(model_type)
                DCT_ALL[model_type] = df_clean
            else:
                DCT_ALL[model_type] = pd.concat([DCT_ALL[model_type], df_clean], ignore_index=True)                    

    # isolate partiulcar setting for qk_share
    df_model = DCT_ALL[[model_type for model_type in list(DCT_ALL.keys()) if fns_manifold in model_type][0]]
    df_model.reset_index(drop=True, inplace=True)
    qk_shares = list(df_model.loc[:,'qk_share'].unique())
    print(qk_shares)
    assert qk_share in qk_shares, f'qk_share = {qk_share} setting does not exist!'
    
    # ---- col names ----
    stats_colnames = ['min', 'max', 'mid', 'median', 'mean', 'std', 'counter']   

    # ----- general settings -----
    num_attention_heads, num_hidden_layers, hidden_size = DCT_ALL[list(DCT_ALL.keys())[0]].loc[0,['n_heads', 'n_layers', 'hidden']]
    #dataset = DCT_ALL[list(DCT_ALL.keys())[0]].loc[0,'dataset_name']
    assert selected_dataset in DCT_ALL[list(DCT_ALL.keys())[0]].loc[:,'dataset_name'].unique(), 'selected_dataset does not exist'

    # ----- fns setting -----
    alphas = sorted(df_model.loc[:,'alpha'].unique())[::-1]  # large to small
    epss = sorted(df_model.loc[:,'bandwidth'].unique())    
    if selected_alphas.lower() == 'none':
        selected_alphas = alphas
    else:
        selected_alphas = [float(selected_alpha) for selected_alpha in str2ls(selected_alphas)]
    #eps = epss[0]
    eps = 1  # hard coded

    # ----- models to plot -----
    fns_model_type = fns_manifold + 'fns' + MODEL_SUFFIX    
    other_model_types = ['dp' + MODEL_SUFFIX, 'sink' + MODEL_SUFFIX]  # 
    model_types_to_plot = [fns_model_type] + other_model_types
            
    nrows, ncols = len(metrics), len(is_ops)     
    # figsize = (3*ncols,3.5*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=(5,4))
    axs = matrixify_axs(axs, nrows, ncols)  # convert axs to 2D array
    # label_axs(fig, axs)  # alphabetically label subfigures             

    model_types_plotted = []
    model_types_seeds = {}
    source_data_path = Path(source_data_path)
    source_data_path.parent.mkdir(parents=True, exist_ok=True)
    for (row_idx, metric), (col_idx, is_op) in product(enumerate(metrics), enumerate(is_ops)):
        source_data = {}
        panel = ascii_lowercase[row_idx * ncols + col_idx]
        ax = axs[row_idx, col_idx] 
        # summary statistics
        row_stats = []

        print(f'model_type = {model_type}')        
        for model_type in model_types_to_plot:
            if is_op:
                model_type = 'op' + model_type
            if model_type in DCT_ALL.keys():
                df_model = DCT_ALL[model_type]
            else:
                continue
            # matching conditions for model setup
            condition0 = (df_model['ensembles']>0)&(df_model['qk_share']==qk_share)&(df_model['is_op']==is_op)&\
                         (df_model['model_dir'].str.contains(selected_dataset))&\
                         (df_model['model_dir'].str.contains(f'/{model_type}-'))
            matching_df = df_model[condition0]

            if model_type not in model_types_plotted:
                model_types_plotted.append(model_type)

            lstyle_model = LINESTYLE_DICT[model_type]
            for alpha in selected_alphas:
                is_fns = 'fns' in model_type
                alpha = alpha if is_fns else None
                matching_df.reset_index(drop=True, inplace=True)                
                                       
                # color
                if is_fns:
                    color = '#2E63A6' if alpha == 1.2 else '#A4292F'
                else:
                    # color = 'k'
                    if 'dpformer' in model_type:
                        color = '#636363'
                    elif 'sinkformer' in model_type:
                        color = 'k'
                # color = HYP_CMAP(HYP_CNORM(alpha)) if is_fns else OTHER_COLORS_DICT[model_type]  
                # -------------------- SINK, DP -------------------- 
                model_info = matching_df 
                # -------------------- FNS --------------------
                if is_fns:
                    # matching conditions for FNS setup
                    condition = (matching_df['alpha']==alpha) & (matching_df['bandwidth']==eps)
                    model_info = model_info[condition]
                # get aggregated training curves
                if model_info.shape[0] > 0:
                    seeds, qk_share = (model_info[k].item() for k in ('seeds', 'qk_share'))                
                    epochs, run_perf_all = load_seed_runs(model_info['model_dir'].item(), seeds, metric)   
                else:
                    continue

                if run_perf_all is not None:
                    counter = run_perf_all.shape[1] - run_perf_all.tail(1).isna().sum(1).item()
                    metric_curves = get_metric_curves(run_perf_all)      
                    if is_fns:
                        plot_label = rf'$\alpha = {alpha}$'
                    elif 'sink' in model_type:
                        plot_label = 'SINK'
                    elif 'dp' in model_type:
                        plot_label = 'DP'
                    exe_plot = ax.plot(epochs, metric_curves[1], linestyle='-', c=color, alpha=1, clip_on=False, label=plot_label)
                    if (row_idx,col_idx) == (0,0):
                        im = exe_plot       
                    # if metric_type == 'mean':
                    # # Calculate std                       
                    # metric_std = np.nanstd(run_perf_all.to_numpy(), axis=1)
                    ax.fill_between(epochs, metric_curves[0], metric_curves[2], color=color, alpha=0.3, clip_on=False, edgecolor='none')                    

                    label = f'alpha={alpha:g}' if is_fns else plot_label
                    source_data[f'{label}_median'] = pd.Series(metric_curves[1].to_numpy(), index=epochs)
                    source_data[f'{label}_band_lower'] = pd.Series(metric_curves[0].to_numpy(), index=epochs)
                    source_data[f'{label}_band_upper'] = pd.Series(metric_curves[2].to_numpy(), index=epochs)

                    # results of the final epoch
                    row_stats.append([model_type, alpha] +\
                                     final_epoch_stats(run_perf_all) + [counter])    
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    # ax.set_xlim([0,20])
                    # ax.set_xticks([0, 5, 10, 15, 20])
                    if row_idx == 0:
                        # ax.set_ylim(bottom=72,top=85)
                        # ax.set_yticks([75,80,85])
                        pass
                    elif row_idx == 1:
                        # ax.set_ylim([0.45, 0.6])
                        # ax.set_yticks([0.45, 0.5, 0.55])
                        pass
                if not is_fns:
                    break  # only do once if model is not FNS type

        summary_stats = pd.DataFrame(data=row_stats, columns=['model_type','alpha']+stats_colnames)

        # print message
        print(metric)
        print(f'is_op = {is_op}, qk_share = {qk_share}')
        print(summary_stats)
        print('\n')                    

        panel_path = f'{source_data_path}{panel}.csv'
        pd.DataFrame(source_data).to_csv(panel_path, index_label='epoch')
        print(f'Source data saved in {panel_path}')

    # # labels
    # model_labels = []
    # for model_type in model_types_plotted:  
    #     if model_type[:2] != 'op': 
    #         color = 'k' if 'fns' in model_type else OTHER_COLORS_DICT[model_type]            
    #         model_label = NAMES_DICT[model_type]
    #         if model_label not in model_labels:            
    #             axs[0,0].plot([], [], c=color, linestyle=LINESTYLE_DICT[model_type], label=model_label)
    #             model_labels.append(model_label)

    # # legend
    axs[0,0].legend(loc='best', frameon=False, ncols=2)                     
    # for alpha in selected_alphas[::-1]:
    #     axs[0,0].plot([], [], c=HYP_CMAP(HYP_CNORM(alpha)), linestyle='solid', 
    #                   label=rf'$\alpha$ = {alpha}')         
    # ncol_legend = 2  #if len(model_types_plotted) == 3 else 1
    # if len(model_types_plotted) >= 2:
    #     #axs[0,0].legend(loc='best', ncol=ncol_legend, frameon=False)           
    #     axs[0,0].legend(loc='best', ncol=ncol_legend, frameon=False)                     

    # Add shared x and y labels     
    #fig.supxlabel('Epochs', fontsize='medium'); fig.supylabel(NAMES_DICT[metrics[0]], fontsize='medium')

    for row_idx in range(len(metrics)):        
        for col_idx, is_op in enumerate(is_ops):  
            ax = axs[row_idx, col_idx]
            #ax.set_ylabel(NAMES_DICT[metric])
            if row_idx == 0:
                #ax.set_title(NAMES_DICT[metric])
                ax_title = r'$\mathbf{W}_{Q,K} \in O(d)$' if is_ops[col_idx] else r'$\mathbf{W}_{Q,K} \notin O(d)$'
                ax.set_title(ax_title)
            
            axs[row_idx,col_idx].sharey(axs[row_idx, 0])
            axs[-1,col_idx].set_xlabel('Epochs')
        # axs[row_idx,0].set_ylabel(NAMES_DICT[metrics[row_idx]])
    axs[0,0].set_ylabel('Testing accuracy (%)')
    axs[1,0].set_ylabel('Testing loss')

    # subfigure labels
    for ii, ax in enumerate(axs.flatten()):
        ax.text(-0.1, 1.13, rf"$\mathbf{{{ascii_lowercase[ii]}}}$",
            transform=ax.transAxes, ha='left',  va='top',
            usetex=False)

    # Adjust layout
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.tight_layout()
    
    dataset_name_short = ''
    if isinstance(selected_dataset,str):
        if '_' in selected_dataset:
            for s in selected_dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += selected_dataset

    model_types_short = [model_type.replace(MODEL_SUFFIX,'') for model_type in model_types_plotted]

    from constants import FIGS_DIR
    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if display:
        plt.show()
    else:
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
        fig_file = models_root.split('/')[1] + '-'
        #fig_file += f'layers={num_hidden_layers}-heads={num_attention_heads}-hidden={hidden_size}-'            
        fig_file += f'l={num_hidden_layers}-d={hidden_size}-'
        fig_file += 'qqv-' if qk_share else 'qkv-'
        fig_file += '_'.join(model_types_short)+ '-' + metrics[0] + '-' + f'ds={dataset_name_short}'
        fig_file += '.pdf'
        plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')            
        print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')

    # separate colorbar
    if cbar_separate:    
        """
        #fig.subplots_adjust(right=0.8)
        fig = plt.figure()
        cbar_ax = fig.add_axes([0.85, 0.20, 0.03, 0.75])
        cbar_ticks = list(np.arange(1,2.01,0.2))
        cbar = fig.colorbar(im, cax=cbar_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticks)
        cbar.ax.tick_params(axis='y', labelsize=tick_size)
        """
        
        fig = plt.figure()
        cbar_ax = fig.add_axes([0.85, 0.20, 0.03, 0.75])
        cbar_ticks = list(np.linspace(1,2,6))
        
        cbar = mpl.colorbar.ColorbarBase(cbar_ax, norm=HYP_CNORM, cmap=HYP_CM)
        cbar.ax.set_yticklabels(cbar_ticks)
        cbar.ax.tick_params(axis='y', labelsize=16.5)

        plt.savefig(njoin(SAVE_DIR,"alpha_colorbar.pdf"), bbox_inches='tight')  


# python plot_results.py trainset_pct_effects_v2 .droot
def trainset_pct_effects_v2(models_root, source_data_path='../.source/edfig3',
                            trainset_pcts='0.5,0.4,0.3,0.2,0.1',
                            fns_manifold='rd', is_rescale_dist=True,
                            qk_shares=[False, True], selected_alphas='1.2,2',
                            metric='val_acc', selected_dataset='imdb', depths=[1],
                            is_op=True, display=False):
    """Plot final metrics against training-set fraction for fixed dimensions.

    ``models_root`` contains directories named
    ``L-d-grid-trainset_pct={pct}``. The shorter ``L-d-g-`` prefix is also
    accepted.

    Rows correspond to the Q/K-sharing settings, columns correspond to fixed
    hidden dimensions, and the x-axis follows the order supplied in
    ``trainset_pcts``.
    Save each panel to <source_data_path><panel>.csv, including plotted errors.
    """
    global trainset_pct_summary_stats

    def normalize_list(value):
        if isinstance(value, str):
            values = str2ls(value)
        elif np.isscalar(value):
            values = [value]
        else:
            values = list(value)

        return [
            item.strip() if isinstance(item, str) else item
            for item in values
        ]

    trainset_pcts = [
        float(pct) for pct in normalize_list(trainset_pcts)
    ]
    if len(trainset_pcts) == 0:
        raise ValueError('trainset_pcts must contain at least one value.')
    if any(
        not np.isfinite(pct) or pct <= 0 or pct >= 1
        for pct in trainset_pcts
    ):
        raise ValueError(
            'Each trainset_pct must be a finite value in (0, 1).'
        )

    qk_shares = [
        str2bool(value) for value in normalize_list(qk_shares)
    ]
    selected_alphas = [
        float(alpha) for alpha in normalize_list(selected_alphas)
    ]
    depths = [
        int(depth) for depth in normalize_list(depths)
    ]

    if (
        len(qk_shares) == 0
        or len(selected_alphas) == 0
        or len(depths) == 0
    ):
        raise ValueError(
            'qk_shares, selected_alphas, and depths cannot be empty.'
        )

    if any(not np.isfinite(alpha) for alpha in selected_alphas):
        raise ValueError(
            'selected_alphas must contain only finite values.'
        )

    if any(depth <= 0 for depth in depths):
        raise ValueError(
            'depths must contain only positive integers.'
        )

    is_op, is_rescale_dist, display = map(
        str2bool,
        (is_op, is_rescale_dist, display)
    )

    if fns_manifold not in ['sphere', 'sp', 'rd', 'v2_rd']:
        raise ValueError(f'{fns_manifold} does not exist!')

    if metric not in [
        'train_acc', 'train_loss', 'val_acc', 'val_loss'
    ]:
        raise ValueError(f'Unsupported metric: {metric}')

    if not isdir(models_root):
        raise FileNotFoundError(
            f'Models root does not exist: {models_root}'
        )

    # Spherical model directories use the "sp" prefix.
    manifold_prefix = (
        'sp' if fns_manifold == 'sphere' else fns_manifold
    )

    fns_type = manifold_prefix + 'fns' + MODEL_SUFFIX
    other_model_types = ['dpformer', 'sinkformer']

    if is_op:
        fns_type = 'op' + fns_type
        other_model_types = [
            'op' + model_type
            for model_type in other_model_types
        ]

    model_specs = [
        {
            'model_type': fns_type,
            'alpha': alpha
        }
        for alpha in selected_alphas
    ] + [
        {
            'model_type': model_type,
            'alpha': None
        }
        for model_type in other_model_types
    ]

    def find_pct_root(pct):
        matches = []
        pattern = re.compile(
            r'L-d-grid-trainset_pct=(.+)'
        )

        if pct == 0.5:
            return njoin(models_root, 'L-d-grid-v1mask-v7scaling-v2')
        else:
            for dirname in os.listdir(models_root):
                match = pattern.fullmatch(dirname)
                directory = njoin(models_root, dirname)

                if match is None or not isdir(directory):
                    continue

                try:
                    directory_pct = float(match.group(1))
                except ValueError:
                    continue

                if math.isclose(
                    directory_pct,
                    pct,
                    rel_tol=0,
                    abs_tol=1e-12
                ):
                    matches.append(dirname)

            if len(matches) == 0:
                pct_text = f'{pct:g}'
                raise FileNotFoundError(
                    f'No L-d-grid-trainset_pct={pct_text} '
                    f'(or L-d-g equivalent) under {models_root}'
                )

            # Prefer the standard L-d-grid naming convention.
            matches.sort(
                key=lambda name: (
                    'L-d-grid-' not in name,
                    name
                )
            )

            return njoin(models_root, matches[0])

    pct_roots = [
        find_pct_root(pct)
        for pct in trainset_pcts
    ]

    layer_pattern = (
        r'(?P<layer>\d+)L-hidden=(?P<hidden>\d+)'
        r'-max_len=512'
    )
    if is_rescale_dist:
        layer_pattern += '-rescaled'

    layer_pattern = re.compile(layer_pattern)

    layer_dirs = {}
    emb_ds = set()

    for pct_idx, pct_root in enumerate(pct_roots):
        for dirname in os.listdir(pct_root):
            match = layer_pattern.fullmatch(dirname)
            layer_dir = njoin(pct_root, dirname)

            if match is None or not isdir(layer_dir):
                continue

            layer = int(match.group('layer'))
            hidden = int(match.group('hidden'))

            if layer not in depths:
                continue

            layer_dirs[pct_idx, layer, hidden] = layer_dir
            emb_ds.add(hidden)

        if not any(
            key[0] == pct_idx
            for key in layer_dirs
        ):
            raise FileNotFoundError(
                f'No matching model directories for '
                f'depths={depths} under {pct_root}'
            )

    emb_ds = np.array(sorted(emb_ds))
    emb_ds = emb_ds[emb_ds < 65]

    final_stats = {}
    summary_rows = []

    def collect_setting_models(layer_dir, qk_share):
        qk_dirname = (
            'config_qqv' if qk_share else 'config_qkv'
        )
        dataset_dir = njoin(
            layer_dir,
            qk_dirname,
            selected_dataset
        )

        if not isdir(dataset_dir):
            return {}

        all_models = {}

        for setting_dir in find_subdirs(
            dataset_dir,
            MODEL_SUFFIX
        ):
            model_dfs = collect_model_dirs(
                setting_dir,
                suffix=MODEL_SUFFIX
            )

            for model_type, model_df in model_dfs.items():
                if model_type in all_models:
                    all_models[model_type] = pd.concat(
                        [
                            all_models[model_type],
                            model_df
                        ],
                        ignore_index=True
                    )
                else:
                    all_models[model_type] = model_df

        return all_models

    def final_seed_values(model_df):
        values = []

        for _, model_row in model_df.iterrows():
            _, seed_runs = load_seed_runs(
                model_row['model_dir'],
                model_row['seeds'],
                metric
            )

            if seed_runs is None:
                continue

            for seed_idx in range(seed_runs.shape[1]):
                seed_values = pd.to_numeric(
                    seed_runs.iloc[:, seed_idx],
                    errors='coerce'
                ).dropna()

                if len(seed_values) == 0:
                    continue

                value = float(seed_values.iloc[-1])

                if np.isfinite(value):
                    values.append(value)

        return np.asarray(values, dtype=float)

    for (pct_idx, layer, hidden), layer_dir in layer_dirs.items():
        for qk_idx, qk_share in enumerate(qk_shares):
            models_all = collect_setting_models(
                layer_dir,
                qk_share
            )

            for model_idx, model_spec in enumerate(model_specs):
                model_type = model_spec['model_type']

                if model_type not in models_all:
                    continue

                model_df = models_all[model_type]

                required_cols = {
                    'ensembles',
                    'qk_share',
                    'is_op',
                    'dataset_name'
                }
                if not required_cols.issubset(model_df.columns):
                    continue

                model_qk_shares = model_df['qk_share'].map(
                    lambda value:
                    str2bool(value)
                    if isinstance(value, str)
                    else bool(value)
                )
                model_is_ops = model_df['is_op'].map(
                    lambda value:
                    str2bool(value)
                    if isinstance(value, str)
                    else bool(value)
                )

                condition = (
                    pd.to_numeric(
                        model_df['ensembles'],
                        errors='coerce'
                    ) > 0
                )
                condition &= model_qk_shares == qk_share
                condition &= model_is_ops == is_op
                condition &= (
                    model_df['dataset_name']
                    == selected_dataset
                )

                alpha = model_spec['alpha']

                if alpha is not None:
                    if (
                        'alpha' not in model_df.columns
                        or 'bandwidth' not in model_df.columns
                    ):
                        continue

                    condition &= np.isclose(
                        pd.to_numeric(
                            model_df['alpha'],
                            errors='coerce'
                        ),
                        alpha,
                        equal_nan=False
                    )
                    condition &= np.isclose(
                        pd.to_numeric(
                            model_df['bandwidth'],
                            errors='coerce'
                        ),
                        1,
                        equal_nan=False
                    )

                values = final_seed_values(
                    model_df[condition]
                )

                if len(values) == 0:
                    continue

                mean = float(np.nanmean(values))
                std = float(np.nanstd(values,ddof=1))

                final_stats[
                    pct_idx,
                    qk_idx,
                    model_idx,
                    layer,
                    hidden
                ] = (mean, std)

                summary_rows.append([
                    trainset_pcts[pct_idx],
                    qk_share,
                    layer,
                    hidden,
                    model_type,
                    alpha,
                    mean,
                    std,
                    float(np.nanmin(values)),
                    float(np.nanmax(values)),
                    int(np.isfinite(values).sum())
                ])

    trainset_pct_summary_stats = pd.DataFrame(
        summary_rows,
        columns=[
            'trainset_pct',
            'qk_share',
            'layer',
            'hidden',
            'model_type',
            'alpha',
            'mean',
            'std',
            'min',
            'max',
            'counter'
        ]
    )

    if trainset_pct_summary_stats.empty:
        raise ValueError(
            'No completed runs matched the requested percentages '
            'and plot settings.'
        )

    # Rows are Q/K settings; columns are fixed hidden dimensions.
    nrows = len(qk_shares)
    ncols = len(emb_ds)

    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(1.8 * ncols, 1.8 * nrows),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    # Categorical positions preserve the supplied percentage order.
    x = np.arange(len(trainset_pcts))
    xlabels = [
        f'{pct:g}'
        for pct in trainset_pcts
    ]

    transparencies = (
        np.linspace(0.55, 1, len(depths))
        if len(depths) > 1
        else [1]
    )

    plot_linestyles = ['-', '--', '-.', ':']

    def model_plot_settings(model_spec):
        alpha = model_spec['alpha']

        if alpha is not None:
            color_idx = round(
                (alpha - 1) / 0.2
            ) + 1

            if 0 <= color_idx < len(COLORS_ALPHA):
                color = COLORS_ALPHA[color_idx]
            else:
                color = HYP_CMAP(
                    HYP_CNORM(alpha)
                )

            return rf'$\alpha={alpha:g}$', color

        if 'dpformer' in model_spec['model_type']:
            # return 'DP', 'k'
            return 'DP', '#636363'

        if 'sinkformer' in model_spec['model_type']:
            # return 'SINK', '#636363'
            return 'SINK', 'k'

        return model_spec['model_type'], '#636363'

    if metric == 'val_acc':
        metric_label = 'Testing accuracy (%)'
    elif metric == 'train_acc':
        metric_label = 'Training accuracy (%)'
    else:
        metric_label = NAMES_DICT.get(metric, metric)

    source_data_path = Path(source_data_path)
    source_data_path.parent.mkdir(parents=True, exist_ok=True)
    for qk_idx, qk_share in enumerate(qk_shares):
        for hidden_idx, hidden in enumerate(emb_ds):
            source_data = {'x': x, 'trainset_pct': trainset_pcts}
            ax = axs[qk_idx, hidden_idx]
            has_curve = False

            for model_idx, model_spec in enumerate(model_specs):
                base_label, color = model_plot_settings(
                    model_spec
                )

                for depth_idx, depth in enumerate(depths):
                    means = []
                    stds = []

                    for pct_idx in range(len(trainset_pcts)):
                        mean, std = final_stats.get(
                            (
                                pct_idx,
                                qk_idx,
                                model_idx,
                                depth,
                                hidden
                            ),
                            (np.nan, np.nan)
                        )
                        means.append(mean)
                        stds.append(std)

                    means = np.asarray(means)
                    stds = np.asarray(stds)

                    if np.all(np.isnan(means)):
                        continue

                    label = base_label
                    if len(depths) > 1:
                        label += rf' $(L={depth})$'

                    ax.errorbar(
                        x,
                        means,
                        yerr=stds,
                        fmt='.',
                        linestyle=plot_linestyles[
                            depth_idx
                            % len(plot_linestyles)
                        ],
                        label=label,
                        c=color,
                        alpha=transparencies[depth_idx],
                        clip_on=False,
                        zorder=1
                    )
                    source_label = label.replace('$', '').replace(r'\alpha', 'alpha').replace(' ', '')
                    source_data[f'{source_label}_mean'] = means
                    source_data[f'{source_label}_std'] = stds
                    has_curve = True

            panel_idx = qk_idx * ncols + hidden_idx
            panel = ascii_lowercase[panel_idx] if panel_idx < 26 else str(panel_idx + 1)
            panel_path = f'{source_data_path}{panel}.csv'
            pd.DataFrame(source_data).to_csv(panel_path, index=False)
            print(f'Source data saved in {panel_path}')

            if not has_curve:
                ax.text(
                    0.5,
                    0.5,
                    'No matching runs',
                    transform=ax.transAxes,
                    ha='center',
                    va='center',
                    color='0.45'
                )

            ax.set_xticks(x)
            ax.set_xticklabels(xlabels)
            ax.set_yticks(list(range(55, 86, 10)))

            # sharey=True normally suppresses labels outside column zero.
            ax.tick_params(
                axis='y',
                labelleft=True
            )

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if qk_idx == 0:
                ax.set_title(
                    rf'$d={hidden}$'
                )

            if qk_idx == nrows - 1:
                ax.set_xlabel(
                    r'$p_{\rm train}$'
                )

            if hidden_idx == 0:
                # qk_label = (
                #     r'$\mathbf{Q} = \mathbf{K}$'
                #     if qk_share
                #     else r'$\mathbf{Q} \neq \mathbf{K}$'
                # )
                # ax.set_ylabel(
                #     qk_label + '\n' + metric_label
                # )
                ax.set_ylabel(
                    metric_label
                )

    # Combine legend entries across all panels in case some models are missing
    # from the first panel.
    legend_by_label = {}

    for ax in axs.flat:
        handles, labels = ax.get_legend_handles_labels()

        for handle, label in zip(handles, labels):
            legend_by_label.setdefault(
                label,
                handle
            )

    if legend_by_label:
        fig.legend(
            list(legend_by_label.values()),
            list(legend_by_label.keys()),
            frameon=False,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.05),
            ncol=len(legend_by_label)
        )

    if axs.size <= len(ascii_lowercase):
        for panel_idx, ax in enumerate(axs.flat):
            ax.text(
                -0.1,
                1.16,
                rf'$\mathbf{{{ascii_lowercase[panel_idx]}}}$',
                transform=ax.transAxes,
                ha='left',
                va='top',
                usetex=False
            )

    print(metric)
    pd.set_option('display.max_rows', None)
    print(trainset_pct_summary_stats.round(3))
    print('\n')

    # Reserve space below the axes for the centered one-row legend.
    plt.tight_layout(
        rect=[0, 0.1, 1, 1]
    )

    save_dir = njoin(
        FIGS_DIR,
        'nlp-task'
    )

    if display:
        plt.show()
    else:
        if not isdir(save_dir):
            makedirs(save_dir)

        fig_file = (
            'L-d-grid-trainset_pct_effects_v2.pdf'
        )
        figure_path = njoin(
            save_dir,
            fig_file
        )

        plt.savefig(
            figure_path,
            bbox_inches='tight'
        )
        print(
            f'Figure saved in {figure_path}'
        )

    return fig, axs


"""
python plot_results.py window_effects .droot/full_models-v1mask-v7scale/
"""
def window_effects(models_root, source_data_path='../.source_data/edfig4',
                   seq_lens=[128, 256, 512, 1024],
                   selected_dataset='imdb', qk_share=False,
                   is_ops='False,True', fns_manifold='rd',
                   selected_alphas='1.2,2', metrics='val_acc,val_loss',
                   display=False):
    """Plot context-window effects; save each panel to <source_data_path><panel>.csv."""
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    global qk_shares, summary_stats, run_perf_all, DCT_ALL

    assert fns_manifold in ['sp', 'rd', 'v2_rd'], f'{fns_manifold} does not exist!'
    qk_share, display = map(str2bool, (qk_share, display))
    seq_lens = [int(seq_len) for seq_len in str2ls(seq_lens)]
    metrics, is_ops = str2ls(metrics), str2ls(is_ops)
    is_ops = [str2bool(is_op) for is_op in is_ops]

    # collect subdirs containing the model directories
    model_root_dirs = find_subdirs(models_root, MODEL_SUFFIX)
    print(model_root_dirs)

    # all trained model types
    model_types = []
    DCT_ALL = {}
    for model_root_dir in model_root_dirs:
        DCT_cur = collect_model_dirs(model_root_dir, suffix=MODEL_SUFFIX)
        for model_type, df_model_cur in DCT_cur.items():
            df_clean = df_model_cur.dropna(subset='alpha') if 'alpha' in df_model_cur.columns else df_model_cur
            if model_type not in DCT_ALL:
                model_types.append(model_type)
                DCT_ALL[model_type] = df_clean
            else:
                DCT_ALL[model_type] = pd.concat([DCT_ALL[model_type], df_clean], ignore_index=True)

    # isolate particular setting for qk_share
    fns_keys = [model_type for model_type in list(DCT_ALL.keys()) if fns_manifold in model_type]
    assert len(fns_keys) > 0, f'{fns_manifold} setting does not exist!'
    df_model = DCT_ALL[fns_keys[0]].copy()
    df_model.reset_index(drop=True, inplace=True)
    qk_shares = list(df_model.loc[:, 'qk_share'].unique())
    print(qk_shares)
    assert qk_share in qk_shares, f'qk_share = {qk_share} setting does not exist!'

    # ---- col names ----
    stats_colnames = ['min', 'max', 'mid', 'median', 'mean', 'std', 'counter']

    # ----- general settings -----
    assert selected_dataset in pd.concat([
        df.loc[:, 'dataset_name'] for df in DCT_ALL.values()
        if 'dataset_name' in df.columns
    ]).unique(), 'selected_dataset does not exist'

    # ----- fns setting -----
    alphas = sorted(df_model.loc[:, 'alpha'].dropna().unique())[::-1]  # large to small
    if isinstance(selected_alphas, str) and selected_alphas.lower() == 'none':
        selected_alphas = alphas
    else:
        selected_alphas = [float(selected_alpha) for selected_alpha in str2ls(selected_alphas)]
    eps = 1  # hard coded

    # ----- models to plot -----
    fns_model_type = fns_manifold + 'fns' + MODEL_SUFFIX
    other_model_types = ['dp' + MODEL_SUFFIX, 'sink' + MODEL_SUFFIX]
    model_types_to_plot = [fns_model_type] + other_model_types

    nrows, ncols = len(metrics), len(is_ops)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5, 4))
    axs = matrixify_axs(axs, nrows, ncols)

    def final_value_stats(values):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return None
        metric_min = np.nanmin(values)
        metric_max = np.nanmax(values)
        metric_mid = (metric_min + metric_max) / 2
        metric_median = np.nanmedian(values)
        metric_mean = np.nanmean(values)
        # metric_std = np.nanstd(values)
        metric_std = np.nanstd(values,ddof=1)
        counter = len(values)
        return [metric_min, metric_max, metric_mid, metric_median, metric_mean, metric_std, counter]

    summary_rows = []
    model_types_plotted = []
    source_data_path = Path(source_data_path)
    source_data_path.parent.mkdir(parents=True, exist_ok=True)
    for (row_idx, metric), (col_idx, is_op) in product(enumerate(metrics), enumerate(is_ops)):
        ax = axs[row_idx, col_idx]
        source_data = {'seq_len': pd.Series(seq_lens, index=np.arange(1, len(seq_lens) + 1))}
        row_stats = []

        for model_type in model_types_to_plot:
            if is_op:
                model_type = 'op' + model_type
            if model_type in DCT_ALL.keys():
                df_model = DCT_ALL[model_type]
            else:
                continue

            # seq_len is the context window length from config.json, fixed before training.
            condition0 = (df_model['ensembles'] > 0) &\
                         (df_model['qk_share'] == qk_share) &\
                         (df_model['is_op'] == is_op) &\
                         (df_model['dataset_name'] == selected_dataset) &\
                         (df_model['seq_len'].isin(seq_lens))
            matching_df = df_model[condition0]

            if model_type not in model_types_plotted:
                model_types_plotted.append(model_type)

            for alpha in selected_alphas:
                is_fns = 'fns' in model_type
                alpha = alpha if is_fns else None

                if is_fns:
                    color = '#2E63A6' if alpha == 1.2 else '#A4292F'
                elif 'dpformer' in model_type:
                    color = '#636363'
                elif 'sinkformer' in model_type:
                    color = 'k'
                else:
                    color = '#636363'

                model_info = matching_df
                if is_fns:
                    condition = (matching_df['alpha'] == alpha) & (matching_df['bandwidth'] == eps)
                    model_info = model_info[condition]

                xs, means, stds = [], [], []
                for seq_len in seq_lens:
                    final_values = []
                    for _, model_row in model_info[model_info['seq_len'] == seq_len].iterrows():
                        _, run_perf_all = load_seed_runs(model_row['model_dir'], model_row['seeds'], metric)
                        if run_perf_all is not None:
                            final_values.extend(run_perf_all.tail(1).to_numpy().ravel().tolist())

                    stats = final_value_stats(final_values)
                    if stats is None:
                        continue

                    xs.append(seq_len)
                    means.append(stats[4])
                    stds.append(stats[5])
                    row_stats.append([seq_len, model_type, alpha] + stats)

                if len(xs) > 0:
                    if is_fns:
                        plot_label = rf'$\alpha = {alpha}$'
                    elif 'sink' in model_type:
                        plot_label = 'SINK'
                    elif 'dp' in model_type:
                        plot_label = 'DP'

                    xs = np.asarray(xs)
                    xs_plot = np.arange(1,len(xs)+1)
                    means = np.asarray(means)
                    stds = np.asarray(stds)
                    ax.plot(xs_plot, means, marker='o', markersize=MARKERSIZE,
                            linestyle='-', c=color, alpha=1, clip_on=False,
                            label=plot_label)
                    ax.fill_between(xs_plot, means - stds, means + stds,
                                    color=color, alpha=0.3, clip_on=False,
                                    edgecolor='none')

                    source_label = f'alpha={alpha:g}' if is_fns else plot_label
                    source_data[f'{source_label}_mean'] = pd.Series(means, index=xs_plot)
                    source_data[f'{source_label}_band_lower'] = pd.Series(means - stds, index=xs_plot)
                    source_data[f'{source_label}_band_upper'] = pd.Series(means + stds, index=xs_plot)

                    ax.set_xticks(xs_plot)
                    ax.set_xticklabels(seq_lens)

                if not is_fns:
                    break  # only do once if model is not FNS type
                
        if row_idx == 0:
            # ax.set_yticks([81, 83, 85, 87])
            ax.set_yticks([75, 79, 83, 87])
            ax.set_ylim(bottom=74)
        elif row_idx == 1:
            # ax.set_yticks([0.43, 0.45, 0.47, 0.49, 0.51])
            ax.set_yticks([0.44, 0.48, 0.52, 0.56])
            ax.set_ylim(top=0.57)

        summary_stats_cur = pd.DataFrame(
            data=row_stats,
            columns=['seq_len', 'model_type', 'alpha'] + stats_colnames
        )
        for row_stat in row_stats:
            summary_rows.append([metric, is_op, qk_share] + row_stat)

        # print message
        print(metric)
        print(f'is_op = {is_op}, qk_share = {qk_share}')
        print(summary_stats_cur)
        print('\n')

        panel = ascii_lowercase[row_idx * ncols + col_idx]
        panel_path = f'{source_data_path}{panel}.csv'
        pd.DataFrame(source_data).to_csv(panel_path, index_label='x')
        print(f'Source data saved in {panel_path}')

    if axs[0, 0].get_legend_handles_labels()[0]:
        axs[0, 0].legend(loc='best', frameon=False, ncols=2)

    for row_idx, metric in enumerate(metrics):
        for col_idx, is_op in enumerate(is_ops):
            ax = axs[row_idx, col_idx]
            if row_idx == 0:
                ax_title = r'$\mathbf{W}_{Q,K} \in O(d)$' if is_ops[col_idx] else r'$\mathbf{W}_{Q,K} \notin O(d)$'
                ax.set_title(ax_title)
            axs[row_idx, col_idx].sharey(axs[row_idx, 0])
            axs[-1, col_idx].set_xlabel('Training context window')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        if 'acc' in metric:
            axs[row_idx, 0].set_ylabel('Testing accuracy (%)')
        elif 'loss' in metric:
            axs[row_idx, 0].set_ylabel('Testing loss')
        else:
            axs[row_idx, 0].set_ylabel(NAMES_DICT.get(metric, metric))

    # subfigure labels
    for ii, ax in enumerate(axs.flatten()):
        ax.text(-0.1, 1.13, rf'$\mathbf{{{ascii_lowercase[ii]}}}$',
                transform=ax.transAxes, ha='left', va='top', usetex=False)

    summary_stats = pd.DataFrame(
        data=summary_rows,
        columns=['metric', 'is_op', 'qk_share', 'seq_len', 'model_type', 'alpha'] + stats_colnames
    )

    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.tight_layout()

    dataset_name_short = ''
    if isinstance(selected_dataset, str):
        if '_' in selected_dataset:
            for s in selected_dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += selected_dataset

    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if display:
        plt.show()
    else:
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
        root_name = Path(models_root).name
        fig_file = f'{root_name}-'
        fig_file += 'qqv-' if qk_share else 'qkv-'
        fig_file += f'max_len_effects-{metrics[0]}-ds={dataset_name_short}.pdf'
        plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')
        print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')

    return fig, axs, summary_stats

# for investigatnig the effects of embedding dim and model depth
"""
python plot_results.py hyperparam_effects .droot/L-d-grid-v1mask-v7scaling-v2/
"""
def hyperparam_effects(models_root, fns_manifold='rd', is_rescale_dist=True,
                       qk_shares=[False, True], selected_alphas='1.2,2',
                       metric='val_acc', selected_dataset='imdb', depths=[1],
                       is_op=True, source_data_path='../.source_data/fig3'):
    """Save each panel's means and plotted standard deviations to prefix + panel + .csv."""

    # PROCESSING
    global metric_matrix, counter_matrix, nan_counter_matrix, average_metric_matrix, run_perf_all
    global other_model_type, fns_type, layers

    linestyles = ['-', '--', '-.', ':']
    markers = ['o', '8', 'p', 's', 'v']

    assert fns_manifold in ['sphere', 'rd', 'v2_rd'], f'{fns_manifold} does not exist!'
    assert metric in ['train_acc', 'train_loss', 'val_acc', 'val_loss']    
    is_op, is_rescale_dist = str2bool(is_op), str2bool(is_rescale_dist)
    fns_type = fns_manifold + 'fns' + MODEL_SUFFIX 
    # other_model_type = 'dpformer'
    other_model_types = ['dpformer', 'sinkformer']
    if is_op:
        fns_type = 'op' + fns_type
        other_model_types = ['op' + other_model_type for other_model_type in other_model_types]
    model_types_to_plot = [fns_type] + other_model_types

    qk_shares = str2ls(qk_shares)        
    selected_alphas = [float(selected_alpha) for selected_alpha in str2ls(selected_alphas)]
    other_model_type_to_idx = {
        model_type: len(selected_alphas) + other_idx
        for other_idx, model_type in enumerate(other_model_types)
    }
    idx_to_other_model_type = {
        model_idx: model_type
        for model_type, model_idx in other_model_type_to_idx.items()
    }

    def other_model_plot_settings(model_type):
        if 'dpformer' in model_type:
            # return 'DP', 'k'
            return 'DP', '#636363'
        if 'sinkformer' in model_type:
            # return 'SINK', '#636363'
            return 'SINK', 'k'
        return model_type, '#636363'

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
    layers = np.array(sorted(list(set(layers)))); layers = layers[layers < 4]
    # emb_ds = np.array(sorted(list(set(emb_ds)))); emb_ds = emb_ds[emb_ds < 65]
    emb_ds = np.array(sorted(list(set(emb_ds)))); emb_ds = emb_ds[emb_ds < 257]
    # emb_ds = np.array(sorted(list(set(emb_ds)))); emb_ds = emb_ds[emb_ds < 513]
    
    #nrows, ncols = len(qk_shares), len(selected_alphas)
    nrows, ncols = len(qk_shares), len(layers)

    # (model_types, qk_share, L, d_model)
    N_model_types = len(selected_alphas) + len(other_model_types)
    average_metric_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
    std_metric_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
    max_metric_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
    min_metric_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
    average_metric_matrix[:] = np.nan
    std_metric_matrix[:] = np.nan
    max_metric_matrix[:] = np.nan
    min_metric_matrix[:] = np.nan
    counter_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
    nan_counter_matrix = np.zeros([nrows, N_model_types, len(layers), len(emb_ds)])
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
            matching_df = df_model[(df_model['ensembles']>0)&(df_model['qk_share']==qk_share)&
                                   (df_model['is_op']==is_op)&                                    
                                   (df_model['model_dir'].str.contains(selected_dataset))&
                                   (df_model['model_dir'].str.contains(f'/{model_type}-'))]

            if model_type not in model_types_plotted:
                model_types_plotted.append(model_type)
            lstyle_model = LINESTYLE_DICT[model_type]            
            for alpha_idx, alpha in enumerate(selected_alphas):  
                # if is fns type
                is_fns = 'fns' in model_type
                alpha = alpha if is_fns else None
                # -------------------- SINK, DP -------------------- 
                model_info = matching_df 
                # -------------------- FNS --------------------
                if is_fns:
                    condition = (matching_df['alpha']==alpha) & (matching_df['bandwidth']==eps)
                    model_info = model_info[condition]
                else:
                    alpha_idx = other_model_type_to_idx[model_type]
                
                if model_info.shape[0] > 0:
                    seeds, qk_share = (model_info[k].item() for k in ('seeds', 'qk_share'))                
                    epochs, run_perf_all = load_seed_runs(model_info['model_dir'].item(), seeds, metric)   
                else:
                    continue

                if run_perf_all is not None:
                    metric_curves = get_metric_curves(run_perf_all)  

                if run_perf_all is not None:
                    average_metric_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        np.nanmean(run_perf_all.loc[run_perf_all.index[-1]:,metric])
                        #np.nanmedian(run_perf_all.loc[run_perf_all.index[-1]:,metric])                                                
                    std_metric_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        np.nanstd(run_perf_all.loc[run_perf_all.index[-1]:,metric],ddof=1)
                    max_metric_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        np.nanmax(run_perf_all.loc[run_perf_all.index[-1]:,metric])
                    min_metric_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        np.nanmin(run_perf_all.loc[run_perf_all.index[-1]:,metric])
                    counter_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        (~np.isnan(run_perf_all.loc[run_perf_all.index[-1]:,metric].to_numpy())).sum()                        
                    nan_counter_matrix[qk_ii,alpha_idx,layer_idx,emb_d_idx] =\
                        (np.isnan(run_perf_all.loc[run_perf_all.index[-1]:,metric].to_numpy())).sum()

                if not is_fns:
                    break  # only do once if model is NOT FNS type                            
    
    # PLOTTING (Just the two I think are most relevant)
    fig, axs = plt.subplots(1,2,figsize=(5, 2),sharex=True)  # ,sharey=True
    
    source_data_path = Path(source_data_path)
    source_data_path.parent.mkdir(parents=True, exist_ok=True)
    source_data = [
        {'x': np.arange(1, len(emb_ds) + 1), 'hidden_size': emb_ds}
        for _ in axs
    ]
    ax = axs[0]
    ax.set_title(r'$\mathbf{Q} \neq \mathbf{K}$')

    if len(depths) == 1:
        trans = [1]
    else:
        trans = [0.5, 1]

    N_model_types = len(selected_alphas) + len(other_model_types)
    for aidx in range(N_model_types):
        for lidx, l in enumerate(depths):
            average_metrics = average_metric_matrix[0,aidx,l-1]
            std_metrics = std_metric_matrix[0,aidx,l-1]
            is_fns = aidx < len(selected_alphas)
            # color                                 
            if is_fns:
                alpha = selected_alphas[aidx]
                legend_label = rf'$\alpha={selected_alphas[aidx]}$'
                # if (alpha == 1.2):
                #     color = '#2E63A6' if l == 2 else '#5391BF'
                # elif (alpha == 2.0):
                #     color = '#A4292F' if l == 2 else '#C86653'      
                color = COLORS_ALPHA[round((alpha - 1)/0.2) + 1]
                #transparency = 1 - 1/(l+1)
                #transparency = l/(l+1)
                #transparency = 1 - np.exp(-l)
                transparency = trans[lidx]                     
                model_type = fns_type       
            elif aidx in idx_to_other_model_type:
                model_type = idx_to_other_model_type[aidx]
                legend_label, color = other_model_plot_settings(model_type)
                transparency = trans[lidx]
                #
                # color = 'k' if l == 2 else '#636363'
                # color = 'k'
            else:
                continue
            linestyle = (0, (2,1)) if l == 2 else '-'
            # X = np.array([1,2,3,4])
            X = np.arange(1,len(emb_ds)+1)
            if np.all(np.isnan(average_metrics)):
                continue
            if len(depths) > 1:
                legend_label = legend_label + r' $(L={})$'.format(l)
            ax.errorbar(X, average_metrics, yerr=std_metrics, 
                        fmt='.', linestyle=linestyle, 
                        label=legend_label, 
                        c=color, alpha=transparency, clip_on=False)
            source_label = legend_label.replace('$', '').replace(r'\alpha', 'alpha').replace(' ', '')
            source_data[0][f'{source_label}_mean'] = average_metrics
            source_data[0][f'{source_label}_std'] = std_metrics

    ax = axs[1]
    ax.set_title(r'$\mathbf{Q} = \mathbf{K}$')
    average_metrics = average_metric_matrix[1,0,0]
    std_metrics = std_metric_matrix[1,0,0]
    for aidx in range(N_model_types):
        for lidx, l in enumerate(depths):
            average_metrics = average_metric_matrix[1,aidx,l-1]
            std_metrics = std_metric_matrix[1,aidx,l-1]
            is_fns = aidx < len(selected_alphas)
            # color      
            if is_fns:
                alpha = selected_alphas[aidx]
                legend_label = rf'$\alpha={selected_alphas[aidx]}$'
                # if (alpha == 1.2):
                #     color = '#2E63A6' if l == 2 else '#5391BF'
                # elif (alpha == 2.0):
                #     color = '#A4292F' if l == 2 else '#C86653'   
                color = COLORS_ALPHA[round((alpha - 1)/0.2) + 1]
                # transparency = 1 - 1/(l+1)  
                transparency = trans[lidx] 
                model_type = fns_type       
            elif aidx in idx_to_other_model_type:
                model_type = idx_to_other_model_type[aidx]
                legend_label, color = other_model_plot_settings(model_type)
                transparency = trans[lidx]
                # color = 'k' if l == 2 else '#636363'
            else:
                continue
            linestyle = (0, (2,1)) if l == 2 else '-'
            # X = np.array([1,2,3,4])
            X = np.arange(1,len(emb_ds)+1)
            if np.all(np.isnan(average_metrics)):
                continue
            if len(depths) > 1:
                legend_label = legend_label + r' $(L={})$'.format(l)
            ax.errorbar(X, average_metrics, yerr=std_metrics, 
                        fmt='.', linestyle=linestyle, label=legend_label, 
                        c=color, alpha=transparency, clip_on=False)

            source_label = legend_label.replace('$', '').replace(r'\alpha', 'alpha').replace(' ', '')
            source_data[1][f'{source_label}_mean'] = average_metrics
            source_data[1][f'{source_label}_std'] = std_metrics

    for panel_idx, panel_data in enumerate(source_data):
        panel_path = f'{source_data_path}{ascii_lowercase[panel_idx]}.csv'
        pd.DataFrame(panel_data).to_csv(panel_path, index=False)
        print(f'Source data saved in {panel_path}')

    for i, ax in enumerate(axs):
        # ax.set_xticks(X)
        ax.set_xticks(np.arange(1,len(emb_ds)+1))
        # ax.set_xticklabels([8, 16, 32, 64])
        ax.set_xticklabels(emb_ds)
        ax.set_xlabel(r'Dimension $d$')
        # ax.set_ylim(top=85)
        # ax.set_ylim(top=90)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0].set_ylim(top=87)
    axs[1].set_ylim(top=87)
    axs[0].set_yticks([75,80,85])
    axs[1].set_yticks([70,75,80,85])
    axs[0].set_ylabel('Testing accuracy (%)')

    # subfigure labels
    for ii, ax in enumerate(axs.flatten()):
        ax.text(-0.1, 1.13, rf"$\mathbf{{{ascii_lowercase[ii]}}}$",
            transform=ax.transAxes, ha='left',  va='top',
            usetex=False)

    axs[1].legend(frameon=False, bbox_to_anchor=(1, 1))

    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')    
    if not isdir(SAVE_DIR): makedirs(SAVE_DIR)
    fig_file = models_root.split('/')[1] + '-'
    fig_file += 'hyperparam_effects.pdf'
    plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')


# for plotting dynamic inference (2 rows)
"""
python plot_results.py dynamic_inference .droot/L-d-grid-v1mask-v7scaling-v2/
"""
def dynamic_inference(models_root, n_layer=1,
                      fns_type='fns', manifold='rd', is_rescale_dist=True, selected_alphas=[1.2, 2.0],
                      is_op=True, qk_share=False, metric='test_acc',
                      batch_size=64, is_dist_based=True):

    global model_dirs, layers, emb_ds, all_model_dirs, other_types, model_dir, fname
    global metrics_dynamic

    # general setting
    batch_size = int(batch_size)
    is_dist_based = str2bool(is_dist_based)    
    fname = 'dist' if is_dist_based else 'prob'
    fname += f'-bs={batch_size}-inference.csv'

    # get layers, emb_ds from regular expression
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
                layer_dirs_dict[f'{n_layer}-{emb_d}'] = njoin(models_root, layer_dir)
            layers.append(layer)
            emb_ds.append(emb_d)
    layers = np.array(sorted(list(set(layers)))); layers = layers[layers < 4]
    emb_ds = np.array(sorted(list(set(emb_ds)))); emb_ds = emb_ds[emb_ds < 65]    
    assert n_layer in layers, f'{n_layer} does not exist!'

    # get all model dirs
    pattern = re.compile(r"model=\d+$")  # seed paths
    all_model_dirs = [str(p) for p in Path(models_root).rglob("*") if p.is_dir() and pattern.search(str(p))]    
    model_dirs = []
    fns_type = manifold + 'fns' + MODEL_SUFFIX
    # other_type = 'dp'+MODEL_SUFFIX
    other_types = ['dp' + MODEL_SUFFIX, 'sink' + MODEL_SUFFIX]
    if is_op:
        fns_type = 'op' + fns_type
        # other_type = 'op' + other_type
        other_types = ['op' + other_type for other_type in other_types]
    print(f'fns_type = {fns_type}')
    print(f'other_types: {other_types} \n')
    # model_types_to_plot = [fns_type, other_type]
    model_types_to_plot = [fns_type] + other_types
    for model_dir in all_model_dirs:
        # is_fns = f'/{fns_type}' in model_dir
        # is_fns = f'{fns_type}' in model_dir
        # is_dp = f'/{other_type}' in model_dir
        # if is_fns:
        if f'{fns_type}' in model_dir:
            for alpha in selected_alphas:
                if f'alpha={float(alpha)}' in model_dir:
                    if model_dir is not None and isfile(njoin(model_dir, fname)):
                        model_dirs.append(model_dir)
        else:
            for other_type in other_types:
                # if f'/{other_type}' in model_dir:
                if f'{other_type}' in model_dir:
                    if model_dir is not None and isfile(njoin(model_dir, fname)):
                        model_dirs.append(model_dir)

    # number of controlled variables
    inference = pd.read_csv(njoin(model_dirs[0], fname))
    controlled_vars = inference.loc[:,'controlled_variable']  # either distance based or probability based
    N_control_var = len(controlled_vars)
    ensembles = 5  # figure out how to extract this

    metrics_dynamic = np.zeros([len(selected_alphas)+len(other_types), 1, 
                                len(emb_ds), N_control_var, ensembles])
    metrics_dynamic[:] = np.nan
    for model_dir in model_dirs:
        # load config
        attn_setup, config, run_performance, train_setting = load_model_files(model_dir)
        if attn_setup['qk_share'] == qk_share:
            seed, model_name = attn_setup['seed'], attn_setup['model_name']
            hidden = config['hidden']
            is_fns = model_name[-9:] == 'fns' + MODEL_SUFFIX
            is_dp = model_name[-8:] == 'dp' + MODEL_SUFFIX
            is_sink = model_name[-10:] == 'sink' + MODEL_SUFFIX
            if is_fns:
                alpha = attn_setup['alpha']
                alpha_idx = selected_alphas.index(alpha)
            elif is_dp:
                alpha_idx = len(selected_alphas)
            elif is_sink:
                alpha_idx = len(selected_alphas) + 1
            print(model_dir)  # delete
            inference = pd.read_csv(njoin(model_dir, fname))
            metrics_dynamic[alpha_idx, 0, list(emb_ds).index(hidden), :, seed] =\
                inference.loc[:,metric]

    if metric[-3:] == 'acc':
        metrics_dynamic *= 100

    # PLOTTING
    fig, axs = plt.subplots(1,4,figsize=(6,1.7),sharex=True,sharey=True)
    
    for didx, alpha_idx in\
          product(range(len(emb_ds)), range(len(selected_alphas)+len(other_types))):
        is_fns = alpha_idx < len(selected_alphas)
        is_dp = alpha_idx == len(selected_alphas)
        is_sink = alpha_idx == len(selected_alphas) + 1
        if is_fns:
            alpha = selected_alphas[alpha_idx]
            # color = HYP_CMAP(HYP_CNORM(alpha))
            color = '#2E63A6' if alpha == 1.2 else '#A4292F'
        elif is_sink:
            # color = OTHER_COLORS_DICT[other_type]
            color = '#636363'
        elif is_dp:
            color = 'k'

        metric_mean = np.nanmean(metrics_dynamic[alpha_idx,0,didx,:,:],-1)
        metric_std = np.nanstd(metrics_dynamic[alpha_idx,0,didx,:,:],-1)
              
        if didx == 0:
            model_label = rf'$\alpha$ = {alpha}' if is_fns else 'DP' if is_dp else 'SINK'
            axs[didx].plot(controlled_vars, metric_mean,
                                markersize=MARKERSIZE, label=model_label,
                                c=color, linestyle=LINESTYLE_DICT[fns_type])  
        else:
            axs[didx].plot(controlled_vars, metric_mean,
                                markersize=MARKERSIZE, 
                                c=color, linestyle=LINESTYLE_DICT[fns_type])  
        # axs[row, col].errorbar(controlled_vars, metric_mean, yerr=metric_std, fmt='.',
        #                     c=color, linestyle=LINESTYLE_DICT[fns_type])  

        # # error bars
        axs[didx].fill_between(controlled_vars,  metric_mean - metric_std, metric_mean + metric_std,
                                    color=color, alpha=0.2, edgecolor='none')                           

        axs[didx].spines['top'].set_visible(False)
        axs[didx].spines['right'].set_visible(False)
        axs[didx].set_title(rf'$d = {emb_ds[didx]}$')

        if not is_dist_based:
            axs[didx].set_xticks([0,0.5,1])
            axs[didx].set_yticks([50, 70, 90])
            axs[didx].set_xlim([0,1])
            axs[didx].set_ylim([48,90])
            axs[didx].set_xlabel(r'$p$')
        else:
            # axs[didx].set_ylim([0.5,0.9])
            axs[didx].set_xticks([1e-16,1e-10,1e-4,1e2])
            axs[didx].set_yticks([40, 55, 70, 85])
            # axs[didx].set_ylim([40,90])
            axs[didx].set_xscale('log')
            axs[didx].set_xlabel('Distance')
            pass

    # legends
    # for alpha_idx, alpha in enumerate(selected_alphas):
    #     c_hyp = HYP_CMAP(HYP_CNORM(alpha))   
    #     axs[0,0].plot([], [], c=c_hyp, linestyle=LINESTYLE_DICT[fns_type],
    #                 label=rf'$\alpha$ = {alpha}')    
    # axs[0,0].plot([],[], c=OTHER_COLORS_DICT[other_type],linestyle=LINESTYLE_DICT[other_type])                      
    fig.legend(frameon=False, bbox_to_anchor=(0.75,0.1), ncol=len(selected_alphas)+len(other_types))
                        
    # control_var_name = 'Distance threshold' if is_dist_based else 'Removal probability'
    # for col in range(2):
    #     axs[0].set_title(rf'$d = {emb_ds[col]}$')
        # axs[0].set_xlabel(control_var_name)
    axs[0].set_ylabel('Testing accuracy (%)')

    # subfigure labels
    for ii, ax in enumerate(axs.flatten()):
        ax.text(-0.1, 1.21, rf"$\mathbf{{{ascii_lowercase[ii]}}}$",
            transform=ax.transAxes, ha='left',  va='top',
            usetex=False)

    # abbreviate dataset_name
    dataset = attn_setup['dataset_name']
    dataset_name_short = ''
    if isinstance(dataset,str):
        if '_' in dataset:
            for s in dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += dataset

    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if not isdir(SAVE_DIR): makedirs(SAVE_DIR)    
    qkv = 'qqv' if qk_share else 'qkv'
    fig_file = f'{n_layer}L-{metric}-'
    if is_dist_based:           
        fig_file += f'dynamic_inference_dist'
    else:
        fig_file += f'dynamic_inference_prob'
    fig_file += f'-{qkv}.pdf'

    plt.tight_layout()
    plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')            
    print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')        


# for plotting dynamic inference (2 rows)
"""
python plot_results.py dynamic_inference_v2 .droot/L-d-grid-v1mask-v7scaling-v2/
"""
def dynamic_inference_v2(models_root, is_dist_based=False,
                         source_data_path='../.source_data/fig4', n_layer=1,
                         fns_type='fns', manifold='rd', is_rescale_dist=True, selected_alphas=[1.2, 2.0],
                         is_op=True, qk_shares=[True,False], metric='test_acc',
                         batch_size=64):
    """Save each panel's curves and shaded bounds to <source_data_path><panel>.csv."""

    global model_dirs, layers, emb_ds, all_model_dirs, other_types, model_dir, fname
    global metrics_dynamic, layer, emb_d, layer_dirs_dict, layer_dir, N_control_var, inference
    global controlled_vars

    # general setting
    batch_size = int(batch_size)
    is_dist_based = str2bool(is_dist_based)    

    # PLOTTING
    nrows, ncols = len(qk_shares), 4
    height = 1.7 if nrows == 1 else 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(6,height))
                            #sharex=True,sharey=True

    source_data_path = Path(source_data_path)
    source_data_path.parent.mkdir(parents=True, exist_ok=True)
    for row, qk_share in enumerate(qk_shares):

        print(f'qk_share = {qk_share}')

        fname = 'dist' if is_dist_based else 'prob'
        fname += f'-bs={batch_size}-inference.csv'

        # get layers, emb_ds from regular expression
        pattern = r"\d+L-hidden=\d+-max_len=512"
        if is_rescale_dist:            
            pattern += "-rescaled"

        # model type
        fns_type = manifold + 'fns' + MODEL_SUFFIX
        # other_type = 'dp'+MODEL_SUFFIX
        other_types = ['dp' + MODEL_SUFFIX, 'sink' + MODEL_SUFFIX]

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
                    layer_dirs_dict[f'{n_layer}-{emb_d}'] = njoin(models_root, layer_dir)
                if layer not in layers:
                    layers.append(layer)
                if emb_d not in emb_ds:
                    emb_ds.append(emb_d)
                # print(f'layers = {layers}')
        layers = np.array(sorted(list(set(layers)))); layers = layers[layers < 2]
        emb_ds = np.array(sorted(list(set(emb_ds)))); emb_ds = emb_ds[emb_ds < 65]    
        assert n_layer in layers, f'{n_layer} does not exist!'

        # get all model dirs
        pattern = re.compile(r"model=\d+$")  # seed paths
        all_model_dirs = [str(p) for p in Path(models_root).rglob("*") if p.is_dir() and pattern.search(str(p))]    
        model_dirs = []

        if is_op:
            fns_type = 'op' + fns_type
            # other_type = 'op' + other_type
            other_types = ['op' + other_type for other_type in other_types]
        print(f'fns_type = {fns_type}')
        print(f'other_types: {other_types} \n')
        # model_types_to_plot = [fns_type, other_type]
        model_types_to_plot = [fns_type] + other_types
        for model_dir in all_model_dirs:
            # is_fns = f'/{fns_type}' in model_dir
            # is_fns = f'{fns_type}' in model_dir
            # is_dp = f'/{other_type}' in model_dir
            # if is_fns:
            if f'{fns_type}' in model_dir:
                for alpha in selected_alphas:
                    if f'alpha={float(alpha)}' in model_dir:
                        if model_dir is not None and isfile(njoin(model_dir, fname)):
                            model_dirs.append(model_dir)
            else:
                for other_type in other_types:
                    # if f'/{other_type}' in model_dir:
                    if f'{other_type}' in model_dir:
                        if model_dir is not None and isfile(njoin(model_dir, fname)):
                            model_dirs.append(model_dir)

        # number of controlled variables
        inference = pd.read_csv(njoin(model_dirs[0], fname))
        inference = inference.dropna()
        controlled_vars = inference.loc[:,'controlled_variable']  # either distance based or probability based
        N_control_var = len(controlled_vars)
        ensembles = 5  # figure out how to extract this


        metrics_dynamic = np.zeros([len(selected_alphas)+len(other_types), 1, 
                                    len(emb_ds), N_control_var, ensembles])
        metrics_dynamic[:] = np.nan
        for model_dir in model_dirs:
            # load config
            attn_setup, config, run_performance, train_setting = load_model_files(model_dir)
            if attn_setup['qk_share'] == qk_share:
                seed, model_name = attn_setup['seed'], attn_setup['model_name']
                hidden = config['hidden']
                is_fns = model_name[-9:] == 'fns' + MODEL_SUFFIX
                is_dp = model_name[-8:] == 'dp' + MODEL_SUFFIX
                is_sink = model_name[-10:] == 'sink' + MODEL_SUFFIX
                if is_fns:
                    alpha = attn_setup['alpha']
                    alpha_idx = selected_alphas.index(alpha)
                elif is_dp:
                    alpha_idx = len(selected_alphas)
                elif is_sink:
                    alpha_idx = len(selected_alphas) + 1
                # print(model_dir)  # DELETE
                inference = pd.read_csv(njoin(model_dir, fname))
                inference = inference.dropna()
                metrics_dynamic[alpha_idx, 0, list(emb_ds).index(hidden), :, seed] =\
                    inference.loc[:,metric]

        if metric[-3:] == 'acc':
            metrics_dynamic *= 100

        
        source_x = 'distance_threshold' if is_dist_based else 'removal_probability'
        source_data = [{source_x: controlled_vars.to_numpy()} for _ in range(ncols)]
        for didx, alpha_idx in\
            product(range(len(emb_ds)), range(len(selected_alphas)+len(other_types))):
            is_fns = alpha_idx < len(selected_alphas)
            is_dp = alpha_idx == len(selected_alphas)
            is_sink = alpha_idx == len(selected_alphas) + 1
            if is_fns:
                alpha = selected_alphas[alpha_idx]
                # color = HYP_CMAP(HYP_CNORM(alpha))
                color = '#2E63A6' if alpha == 1.2 else '#A4292F'
            # elif is_sink:
            elif is_dp:
                # color = OTHER_COLORS_DICT[other_type]
                color = '#636363'
            # elif is_dp:
            elif is_sink:
                color = 'k'

            metric_mean = np.nanmean(metrics_dynamic[alpha_idx,0,didx,:,:],-1)
            metric_std = np.nanstd(metrics_dynamic[alpha_idx,0,didx,:,:],axis=-1,ddof=1)
                
            ax = axs[row,didx]

            if is_dist_based:
                controlled_var_thresh = 1e-5
            else:
                controlled_var_thresh = 0
            if didx == 0 and row == 0:
                model_label = rf'$\alpha$ = {alpha}' if is_fns else 'DP' if is_dp else 'SINK'
                plot_idxs = controlled_vars[controlled_vars >= controlled_var_thresh].index
                ax.plot(controlled_vars[plot_idxs], metric_mean[plot_idxs],
                                    markersize=MARKERSIZE, label=model_label,
                                    c=color, linestyle=LINESTYLE_DICT[fns_type])  
            else:
                plot_idxs = controlled_vars[controlled_vars >= controlled_var_thresh].index
                ax.plot(controlled_vars[plot_idxs], metric_mean[plot_idxs],
                                    markersize=MARKERSIZE, 
                                    c=color, linestyle=LINESTYLE_DICT[fns_type])  
            # ax.errorbar(controlled_vars, metric_mean, yerr=metric_std, fmt='.',
            #                     c=color, linestyle=LINESTYLE_DICT[fns_type])  

            # # error bars
            ax.fill_between(controlled_vars[plot_idxs],  
                            metric_mean[plot_idxs] - metric_std[plot_idxs], 
                            metric_mean[plot_idxs] + metric_std[plot_idxs],
                            color=color, alpha=0.2, edgecolor='none')                           

            source_label = f'alpha={alpha:g}' if is_fns else 'DP' if is_dp else 'SINK'
            source_data[didx][f'{source_label}_mean'] = metric_mean
            source_data[didx][f'{source_label}_band_lower'] = metric_mean - metric_std
            source_data[didx][f'{source_label}_band_upper'] = metric_mean + metric_std

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if row == 0:
                ax.set_title(rf'$d = {emb_ds[didx]}$')

            if not is_dist_based:
                ax.set_xticks([0,0.5,1])
                ax.set_yticks([50, 70, 90])
                ax.set_xlim([0,1])
                ax.set_ylim([48,90])
                if row == nrows - 1:
                    ax.set_xlabel(r'$p$')
            else:
                # ax.set_ylim([0.5,0.9])
                ax.set_xticks([1e-16,1e-10,1e-4,1e2])
                ax.set_yticks([40, 55, 70, 85])
                # ax.set_ylim([40,90])
                ax.set_xscale('log')
                if row == nrows - 1:
                    ax.set_xlabel('Distance')

        for didx, panel_data in enumerate(source_data):
            panel = ascii_lowercase[row * ncols + didx]
            panel_path = f'{source_data_path}{panel}.csv'
            pd.DataFrame(panel_data).to_csv(panel_path, index=False)
            print(f'Source data saved in {panel_path}')

        axs[row,0].set_ylabel('Testing accuracy (%)')

    # legends
    # for alpha_idx, alpha in enumerate(selected_alphas):
    #     c_hyp = HYP_CMAP(HYP_CNORM(alpha))   
    #     axs[0,0].plot([], [], c=c_hyp, linestyle=LINESTYLE_DICT[fns_type],
    #                 label=rf'$\alpha$ = {alpha}')    
    # axs[0,0].plot([],[], c=OTHER_COLORS_DICT[other_type],linestyle=LINESTYLE_DICT[other_type])                      
    fig.legend(frameon=False, bbox_to_anchor=(0.75,0.05), ncol=len(selected_alphas)+len(other_types))
                        
    # control_var_name = 'Distance threshold' if is_dist_based else 'Removal probability'
    # for col in range(2):
    #     axs[0].set_title(rf'$d = {emb_ds[col]}$')
        # axs[0].set_xlabel(control_var_name)

    # subfigure labels
    for ii, ax in enumerate(axs.flatten()):
        ax.text(-0.1, 1.21, rf"$\mathbf{{{ascii_lowercase[ii]}}}$",
            transform=ax.transAxes, ha='left',  va='top',
            usetex=False)

    # abbreviate dataset_name
    dataset = attn_setup['dataset_name']
    dataset_name_short = ''
    if isinstance(dataset,str):
        if '_' in dataset:
            for s in dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += dataset

    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if not isdir(SAVE_DIR): makedirs(SAVE_DIR)    
    qkv = 'qqv' if qk_share else 'qkv'
    fig_file = f'{n_layer}L-{metric}-'
    if is_dist_based:           
        fig_file += f'dynamic_inference_dist'
    else:
        fig_file += f'dynamic_inference_prob'
    # fig_file += f'-{qkv}.pdf'
    fig_file += f'-all.pdf'

    plt.tight_layout()
    plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')            
    print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')        


"""
python -i plot_results.py len_inference .droot/L-d-grid/
"""
def len_inference(models_root, n_layer=6, max_len_adj=1024,
                  fns_type='fns', manifold='rd', is_rescale_dist=True, selected_alphas=[1.2, 2.0],
                  is_op=False, qk_shares=[False], metric='test_acc'):

    global model_dirs, emb_ds, metric_plot, metrics_all, inference, layers, emb_ds

    # general setting
    if metric == 'test_acc':
        fname = f'test_inference-bs=1-len={max_len_adj}.csv'
    elif metric == 'train_acc':
        fname = f'train_inference-bs=1-len={max_len_adj}.csv'

    # get layers, emb_ds from regular expression
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
                layer_dirs_dict[f'{n_layer}-{emb_d}'] = njoin(models_root, layer_dir)
            layers.append(layer)
            emb_ds.append(emb_d)
    layers = np.array(sorted(list(set(layers)))); # layers = layers[layers < 4]
    emb_ds = np.array(sorted(list(set(emb_ds)))); # emb_ds = emb_ds[emb_ds < 65]    
    assert n_layer in layers, f'{n_layer} does not exist!'

    # get all model dirs
    pattern = re.compile(r"model=\d+$")  # seed paths
    all_model_dirs = [str(p) for p in Path(models_root).rglob("*") if p.is_dir() and pattern.search(str(p))]    
    model_dirs = []
    fns_type = manifold + 'fns' + MODEL_SUFFIX
    other_type = 'dp'+MODEL_SUFFIX
    if is_op:
        fns_type = 'op' + fns_type
        other_type = 'op' + other_type
    model_types_to_plot = [fns_type, other_type]
    for model_dir in all_model_dirs:
        is_fns = f'/{fns_type}' in model_dir
        is_dp = f'/{other_type}' in model_dir
        if is_fns:
            # isolate alphas from SELECTED_ALPHAS
            if not any(f'alpha={float(alpha)}' in model_dir for alpha in selected_alphas):
                continue     
        elif is_dp:
            pass
        else:
            continue
        # elif is_dp:
        if model_dir is not None and isfile(njoin(model_dir, fname)):
            model_dirs.append(model_dir)

    # number of controlled variables
    inference = pd.read_csv(njoin(model_dirs[0], fname))
    seq_lens = inference.loc[:,'seq_len']
    _, config, _, _ = load_model_files(model_dir)    
    thresholds = []
    ii = 6
    #while 2**ii <= config['seq_len']:
    while 2**ii <= max_len_adj:
        thresholds.append(2**ii)
        ii += 1    
    # if seq_lens.max() > max_len_adj:
    #     thresholds.append(seq_lens.max())

    ensembles = 5  # figure out how to extract this

    nrows, ncols = len(qk_shares), len(emb_ds)
    figsize = (3.65*ncols,3*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=figsize,sharex=True,sharey=True)  # layout='constrained'
    axs = matrixify_axs(axs, nrows, ncols)
    label_axs(fig, axs)

    metrics_all = np.zeros([2, len(selected_alphas)+1, len(qk_shares), 
                               len(emb_ds), len(thresholds) + 1, ensembles])
    metrics_all[:] = np.nan
    for model_dir in model_dirs:
        # load config
        attn_setup, config, run_performance, train_setting = load_model_files(model_dir)
        seed, model_name, qk_share = attn_setup['seed'], attn_setup['model_name'],\
              attn_setup['qk_share']
        hidden = config['hidden']
        is_fns = model_name[-9:] == 'fns' + MODEL_SUFFIX
        if is_fns:
            alpha = attn_setup['alpha']
            alpha_idx = selected_alphas.index(alpha)
        else:
            alpha_idx = len(selected_alphas)
        #if isfile(njoin(model_dir, fname)):
        inference = pd.read_csv(njoin(model_dir, fname))
        for tidx, threshold in enumerate(thresholds):
            if tidx == 0:
                mask = inference["seq_len"] <= threshold
            else:
                mask = (thresholds[tidx-1] < inference["seq_len"]) & (inference["seq_len"] <= threshold)
            metrics_all[:, alpha_idx, qk_shares.index(qk_share), list(emb_ds).index(hidden), tidx, seed] =\
                [inference.loc[mask, "is_correct"].sum(), mask.sum()]                

        # greater than max_len
        mask = (thresholds[-1] < inference["seq_len"])
        metrics_all[:, alpha_idx, qk_shares.index(qk_share), list(emb_ds).index(hidden), -1, seed] =\
            [inference.loc[mask, "is_correct"].sum(), mask.sum()]                

    # accuracy is count / total
    metric_plot = metrics_all[0,:] / metrics_all[1,:]
    for sidx, didx, alpha_idx in\
          product(range(len(qk_shares)), range(len(emb_ds)), range(len(selected_alphas)+1)):
        
        ax = axs[sidx,didx]

        is_fns = alpha_idx < len(selected_alphas)
        if is_fns:
            alpha = selected_alphas[alpha_idx]
            color = HYP_CMAP(HYP_CNORM(alpha))
        else:
            color = OTHER_COLORS_DICT[other_type]

        metric_mean = np.nanmean(metric_plot[alpha_idx,sidx,didx,:,:] * 100,-1)
        metric_std = np.nanstd(metric_plot[alpha_idx,sidx,didx,:,:]  * 100,-1)
                            
        # add final dummy threshold
        ax.plot(thresholds + [thresholds[-1] * 2], metric_mean,
                            markersize=MARKERSIZE,
                            c=color, linestyle=LINESTYLE_DICT[fns_type])  

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # error bars
        # ax.fill_between(thresholds + [thresholds[-1] * 2],  
        #                 metric_mean - metric_std, metric_mean + metric_std,
        #                 color=color, alpha=1/2)                           

    # legends
    for alpha_idx, alpha in enumerate(selected_alphas):
        c_hyp = HYP_CMAP(HYP_CNORM(alpha))   
        axs[0,0].plot([], [], c=c_hyp, linestyle=LINESTYLE_DICT[fns_type],
                    label=rf'$\alpha$ = {alpha}')    
    axs[0,0].plot([],[], c=OTHER_COLORS_DICT[other_type],linestyle=LINESTYLE_DICT[other_type])                      
                            
    # log x-axis
    axs[0,0].set_xscale('log')                            

    for ncol in range(ncols):
        axs[0,ncol].set_title(rf'$d = {emb_ds[ncol]}$')
        axs[-1,ncol].set_xlabel('Sequence length')        
        # tick labels
        axs[-1,ncol].set_xticks(thresholds + [thresholds[-1] * 2])
        axs[-1,ncol].set_xticklabels(thresholds + [rf'$\leq$'])  # [rf'${thresholds[-1]} \leq$']
        # remove minor ticks
        axs[-1,ncol].xaxis.set_minor_formatter(NullFormatter()) 
        axs[-1,ncol].xaxis.minorticks_off() 
    for nrow in range(nrows):
        axs[nrow,0].set_ylabel(r'$Q = K$' if qk_shares[nrow] else r'$Q \neq K$')
    axs[0,0].legend(frameon=False)
    
    plt.tight_layout(rect=[0, 0, 0.93, 1])   

    # abbreviate dataset_name
    dataset = attn_setup['dataset_name']
    dataset_name_short = ''
    if isinstance(dataset,str):
        if '_' in dataset:
            for s in dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += dataset

    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if not isdir(SAVE_DIR): makedirs(SAVE_DIR)    
    qkv = 'qqv' if qk_share else 'qkv'
    fig_file = f'{n_layer}L-len={max_len_adj}-is_op={is_op}-{metric}-inference.pdf'
    plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')            
    print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')  


"""
python plot_results.py fna_alpha_effects .droot/full_models-v1mask-v7scale/
"""
def fna_alpha_effects(models_root, selected_dataset='imdb', 
                      source_data_path='../.source_data/edfig5',
                      fns_manifold='rd', qk_share=False, selected_alphas='none',
                      bandwidth=1, metric='val_acc', is_ops=[False, True],
                      display=False):
    """Save each panel's means and plotted standard deviations to prefix + panel + .csv."""
    global summary_stats

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    assert fns_manifold in ['sp', 'rd', 'v2_rd'], f'{fns_manifold} does not exist!'
    qk_share, display = map(str2bool, (qk_share, display))
    if isinstance(is_ops, bool):
        is_ops = [is_ops]
    else:
        is_ops = [str2bool(is_op) for is_op in str2ls(is_ops)]
    if isinstance(bandwidth, str) and bandwidth.lower() == 'none':
        bandwidth = None
    elif bandwidth is not None:
        bandwidth = float(bandwidth)

    model_root_dirs = find_subdirs(njoin(models_root), MODEL_SUFFIX)
    print(model_root_dirs)

    DCT_ALL = {}
    for model_root_dir in model_root_dirs:
        DCT_cur = collect_model_dirs(model_root_dir, suffix=MODEL_SUFFIX)
        for model_type, df_model_cur in DCT_cur.items():
            df_clean = df_model_cur.dropna(subset=['alpha']) if 'alpha' in df_model_cur.columns else df_model_cur
            if model_type not in DCT_ALL:
                DCT_ALL[model_type] = df_clean
            else:
                DCT_ALL[model_type] = pd.concat([DCT_ALL[model_type], df_clean], ignore_index=True)

    fns_model_type = fns_manifold + 'fns' + MODEL_SUFFIX
    fns_model_types = [model_type for model_type in [fns_model_type, 'op' + fns_model_type]
                       if model_type in DCT_ALL]
    assert len(fns_model_types) > 0, f'{fns_model_type} setting does not exist!'

    df_fns_all = pd.concat([DCT_ALL[model_type] for model_type in fns_model_types], ignore_index=True)
    qk_shares = list(df_fns_all.loc[:, 'qk_share'].unique())
    assert qk_share in qk_shares, f'qk_share = {qk_share} setting does not exist!'
    assert selected_dataset in df_fns_all.loc[:, 'dataset_name'].unique(), 'selected_dataset does not exist'

    alphas = sorted(df_fns_all.loc[:, 'alpha'].dropna().unique())
    if selected_alphas is None or (isinstance(selected_alphas, str) and selected_alphas.lower() == 'none'):
        selected_alphas = alphas
    else:
        selected_alphas = sorted([float(selected_alpha) for selected_alpha in str2ls(selected_alphas)])

    def normalize_training_seeds(seeds):
        if seeds is None:
            return []
        if isinstance(seeds, str):
            try:
                seeds = literal_eval(seeds)
            except (SyntaxError, ValueError):
                seeds = str2ls(seeds)
        if isinstance(seeds, (int, np.integer)):
            return [int(seeds)]
        if isinstance(seeds, (float, np.floating)):
            return [] if np.isnan(seeds) else [int(seeds)]
        return [int(seed) for seed in list(seeds)]

    def load_seed_final_values(model_dir, seeds):
        rows = []
        for seed in normalize_training_seeds(seeds):
            seed_path = njoin(model_dir, f'model={seed}')
            fpath = njoin(seed_path, 'run_performance.csv')
            if not isfile(fpath):
                fpath = njoin(seed_path, '_run_performance.csv')
                if not isfile(fpath):
                    continue
            run = pd.read_csv(fpath)
            if metric not in run.columns:
                continue
            seed_metric = run[metric].dropna()
            if len(seed_metric) == 0:
                continue
            if 'acc' in metric and seed_metric.iloc[-1] <= 1:
                seed_metric *= 100
            rows.append({'seed': seed, metric: seed_metric.iloc[-1]})
        return pd.DataFrame(rows)

    ncols = len(is_ops)
    fig, axs = plt.subplots(1, ncols, 
                            # figsize=(2.5*ncols, 2.65),
                            figsize=(2.5*ncols, 2.8), 
                            sharex=True, sharey=True,
                            squeeze=False)
    axs = axs[0]
    row_stats = []
    context_windows = [128, 256, 512, 1024]
    window_colors = {
        128: '#0072B2',
        256: '#009E73',
        512: '#E69F00',
        1024: '#CC79A7',
    }
    window_markers = {128: 'o', 256: 's', 512: 'D', 1024: '^'}
    legend_handles = {}

    source_data_path = Path(source_data_path)
    source_data_path.parent.mkdir(parents=True, exist_ok=True)
    source_data = [{} for _ in range(ncols)]
    for col_idx, is_op in enumerate(is_ops):
        ax = axs[col_idx]
        model_type = ('op' if is_op else '') + fns_model_type
        if model_type not in DCT_ALL:
            continue

        df_model = DCT_ALL[model_type]
        condition = (df_model['ensembles'] > 0) & (df_model['qk_share'] == qk_share) &\
                    (df_model['is_op'] == is_op) &\
                    ((df_model['dataset_name'] == selected_dataset) |
                     (df_model['model_dir'].str.contains(selected_dataset, regex=False))) &\
                    (df_model['model_dir'].str.contains(f'{model_type}-', regex=False))
        matching_df = df_model[condition].copy()
        if bandwidth is not None:
            matching_df = matching_df[matching_df['bandwidth'] == bandwidth]

        matching_df = matching_df.assign(
            seq_len=pd.to_numeric(matching_df['seq_len'], errors='coerce')
        )
        for context_window in context_windows:
            window_df = matching_df[matching_df['seq_len'] == context_window]
            plot_rows = []
            for alpha in selected_alphas:
                model_info = window_df[window_df['alpha'] == alpha]
                seed_values_all = []
                for _, model_row in model_info.iterrows():
                    seed_values = load_seed_final_values(model_row['model_dir'], model_row['seeds'])
                    if len(seed_values) > 0:
                        seed_values_all.append(seed_values)

                if len(seed_values_all) == 0:
                    continue

                seed_values_all = pd.concat(seed_values_all, ignore_index=True).dropna(subset=[metric])
                if len(seed_values_all) == 0:
                    continue

                final_values = seed_values_all[metric]
                counter = len(final_values)
                mean = final_values.mean()
                std = final_values.std(ddof=1) if counter > 1 else 0
                sem = std / math.sqrt(counter) if counter > 0 else np.nan
                median = final_values.median()
                min_val = final_values.min()
                max_val = final_values.max()

                row_stats.append([model_type, context_window, alpha, is_op, qk_share, bandwidth,
                                  mean, std, sem, median, min_val, max_val, counter])
                plot_rows.append([alpha, mean, std, counter])

            if len(plot_rows) == 0:
                continue

            plot_df = pd.DataFrame(
                plot_rows, columns=['alpha', 'mean', 'std', 'counter']
            ).sort_values('alpha')
            x = plot_df['alpha'].to_numpy()
            y = plot_df['mean'].to_numpy()
            yerr = plot_df['std'].fillna(0).to_numpy()
            marker_color = window_colors[context_window]
            marker_shape = window_markers[context_window]

            ax.plot(x, y, c=marker_color, linewidth=1, alpha=0.85, zorder=1)
                    # label=rf'$n$ = {context_window}')
            ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor=marker_color,
                        elinewidth=0.8, capsize=2, alpha=0.6, zorder=1)
            scatter = ax.scatter(x, y, marker=marker_shape, c=marker_color,
                                 s=36, edgecolor='white', linewidth=0.5, zorder=2)
            source_label = f'n={context_window}'
            source_data[col_idx][f'{source_label}_mean'] = pd.Series(y, index=x)
            source_data[col_idx][f'{source_label}_std'] = pd.Series(yerr, index=x)
            if context_window not in legend_handles:
                legend_handles[context_window] = scatter

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$\alpha$')
        ax.set_xticks(selected_alphas)
        ax_title = r'$\mathbf{W}_{Q,K} \in O(d)$' if is_op else r'$\mathbf{W}_{Q,K} \notin O(d)$'
        ax.set_title(ax_title)
        ax.tick_params(axis='y', labelleft=True)

        ax.set_yticks([78,82,86])

    for col_idx, panel_data in enumerate(source_data):
        panel_path = f'{source_data_path}{ascii_lowercase[col_idx]}.csv'
        pd.DataFrame(panel_data).to_csv(panel_path, index_label='alpha')
        print(f'Source data saved in {panel_path}')

    # Subfigure labels
    for ii, ax in enumerate(axs.flatten()):
        ax.text(-0.1, 1.1, rf"$\mathbf{{{ascii_lowercase[ii]}}}$",
            transform=ax.transAxes, ha='left',  va='top',
            usetex=False)

    axs[0].set_ylabel('Testing accuracy (%)' if 'acc' in metric else NAMES_DICT.get(metric, metric))
    summary_stats = pd.DataFrame(
        data=row_stats,
        columns=['model_type', 'seq_len', 'alpha', 'is_op', 'qk_share', 'bandwidth',
                 'mean', 'std', 'sem', 'median', 'min', 'max', 'counter']
    )
    print(metric)
    print(f'qk_share = {qk_share}, bandwidth = {bandwidth}')
    print(summary_stats.round(3))
    print('\n')

    if len(legend_handles) > 0:
        plotted_windows = [window for window in context_windows if window in legend_handles]
        fig.legend(
            [legend_handles[window] for window in plotted_windows],
            [rf'$n$ = {window}' for window in plotted_windows],
            # title='Training context window', 
            # loc='upper center',
            # loc='lower center',
            # loc='best',
            # bbox_to_anchor=(0.5, 0.99), 
            # bbox_to_anchor=(0.5,-0.1),
            bbox_to_anchor=(1.17,0.68),
            # ncol=min(len(plotted_windows), 2*ncols),
            ncol=1,
            frameon=False,
        )
        # axs[0].legend(loc='best',
        #     ncol=min(len(plotted_windows), 2*ncols),
        #     frameon=False,
        # )
        plt.tight_layout(rect=[0, 0, 1, 0.78])
        # plt.tight_layout()
    else:
        plt.tight_layout()

    SAVE_DIR = njoin(FIGS_DIR, 'nlp-task')
    if display:
        plt.show()
    else:
        if not isdir(SAVE_DIR): makedirs(SAVE_DIR)

        dataset_name_short = ''
        if isinstance(selected_dataset, str):
            if '_' in selected_dataset:
                for s in selected_dataset.split('_'):
                    dataset_name_short += s[0]
            else:
                dataset_name_short += selected_dataset

        fig_file = Path(models_root).name + '-'
        fig_file += fns_model_type.replace(MODEL_SUFFIX, '') + '-'
        fig_file += 'qqv-' if qk_share else 'qkv-'
        fig_file += f'alpha_effects-{metric}-ds={dataset_name_short}.pdf'
        plt.savefig(njoin(SAVE_DIR, fig_file), bbox_inches='tight')
        print(f'Figure saved in {njoin(SAVE_DIR, fig_file)}')

    return fig, axs, summary_stats


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
