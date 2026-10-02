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
# ------------------------------------------

MARKERSIZE = 4
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
# return lower, central, and upper curves for mean/std or median/quartiles
def get_metric_curves(run_perf_all,type='median'):
    if type == 'mean':
        metric_m = run_perf_all.mean(axis=1)
        metric_std = run_perf_all.std(axis=1, ddof=1).fillna(0.0)
        metric_l = metric_m - metric_std
        metric_u = metric_m + metric_std
    elif type == 'median':
        metric_m = run_perf_all.quantile(0.5,1)
        metric_l = run_perf_all.quantile(0.25,1)
        metric_u = run_perf_all.quantile(0.75,1)
    return [metric_l, metric_m, metric_u]

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
        iters = run['iter'].astype(int)
        positive_iters = iters[iters > 0]
        if len(positive_iters) > 0:
            epochs = iters // int(positive_iters.iloc[0])
        else:
            epochs = np.arange(len(run))
        if 'acc' in metric and run.loc[run.index[-1], metric] <= 1:
            run[metric] *= 100
        seed_run = run[metric].copy()
        seed_run.index = epochs
        seed_run.name = seed
        runs.append(seed_run)
    if len(runs)==0:
        return (None, None)
    else:
        run_perf_all = pd.concat(runs, axis=1).sort_index()
        return run_perf_all.index.to_numpy(), run_perf_all

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
def phase_ensembles(models_root, selected_dataset='cifar10',
                    fns_manifold='rd', qk_share=False, selected_alphas='1.2,2',
                    metrics='val_acc,val_loss',
                    is_ops = [False,True],  # [False,True]
                    cbar_separate=False, display=False):
    global summary_stats, run_perf_all

    start_epoch, end_epoch = 25, 110

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    assert fns_manifold in ['sp', 'rd', 'v2_rd'], f'{fns_manifold} does not exist!'
    qk_share, cbar_separate, display = map(str2bool, (qk_share, cbar_separate, display))
    metrics, is_ops = str2ls(metrics), str2ls(is_ops)

    # additional plot settings
    lwidth = 1

    # collect subdirs containing the model directories
    model_root_dirs = models_roots = find_subdirs(njoin(models_root), MODEL_SUFFIX)
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
    assert qk_share in qk_shares, f'qk_share = {qk_share} setting does not exist!'
    
    # print('df_model')
    # print(df_model)

    # ---- col names ----
    stats_colnames = ['min', 'max', 'mid', 'median', 'mean', 'std', 'counter']
    summary_colnames = ['metric', 'is_op', 'model_type', 'alpha', 'final_epoch'] + stats_colnames
    summary_rows = []

    # ----- general settings -----
    num_attention_heads, num_hidden_layers, hidden_size =\
         DCT_ALL[list(DCT_ALL.keys())[0]].loc[0,['num_attention_heads', 'num_hidden_layers', 'hidden_size']]
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

    def get_other_model_style(model_type):
        base_model_type = model_type[2:] if model_type.startswith('op') else model_type
        if base_model_type.startswith('dp'):
            return '#636363', 'DP'
        if base_model_type.startswith('sink'):
            return 'k', 'SINK'
        return OTHER_COLORS_DICT.get(model_type, '#636363'), NAMES_DICT.get(model_type, base_model_type.upper())
            
    print(f'model_types_to_plot: {model_types_to_plot}')

    nrows, ncols = len(metrics), len(is_ops)     
    # figsize = (3*ncols,3.5*nrows)
    fig, axs = plt.subplots(nrows,ncols,figsize=(5,4))  # , sharex=True
    # axs = matrixify_axs(axs, nrows, ncols)  # convert axs to 2D array
    # label_axs(fig, axs)  # alphabetically label subfigures             

    model_types_plotted = []
    model_types_seeds = {}     
    for (row_idx, metric), (col_idx, is_op) in product(enumerate(metrics), enumerate(is_ops)):
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
                         (df_model['model_dir'].str.contains(f'{model_type}-'))   
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
                    color = '#2E63A6' if alpha < 2 else '#A4292F'
                    # color = HYP_CMAP(HYP_CNORM(alpha))
                    label = rf'$\alpha = {alpha}$'
                else:
                    color, label = get_other_model_style(model_type)
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
                    run_perf_all = run_perf_all.copy()

                    # load_seed_runs checks only the final CSV row when converting
                    # accuracy fractions to percentages. Repair columns with a
                    # trailing NaN by checking all of their valid values here.
                    if 'acc' in metric:
                        for column_idx in range(run_perf_all.shape[1]):
                            valid_values = run_perf_all.iloc[:, column_idx].dropna()
                            if not valid_values.empty and valid_values.between(0, 1).all():
                                run_perf_all.iloc[:, column_idx] *= 100

                    # Epochs are rows and seeds are columns. Limit the former and
                    # discard seed columns that contain no usable values.
                    run_perf_all = run_perf_all.loc[run_perf_all.index < end_epoch]
                    run_perf_all = run_perf_all.dropna(axis=1, how='all')
                    if run_perf_all.empty or run_perf_all.shape[1] == 0:
                        continue

                    # Compare all seeds over their common observed epoch range.
                    # Forward filling handles different logging cadence without
                    # extending any seed beyond its final recorded epoch.
                    first_epochs = [series.first_valid_index() for _, series in run_perf_all.items()]
                    last_epochs = [series.last_valid_index() for _, series in run_perf_all.items()]
                    common_start_epoch = max(first_epochs)
                    common_end_epoch = min(last_epochs)
                    display_start_epoch = max(start_epoch, common_start_epoch)
                    if display_start_epoch > common_end_epoch:
                        continue

                    aligned_run_perf = run_perf_all.ffill()
                    aligned_run_perf = aligned_run_perf.loc[
                        (aligned_run_perf.index >= display_start_epoch)
                        & (aligned_run_perf.index <= common_end_epoch)
                    ]
                    if aligned_run_perf.empty:
                        continue

                    # Use the same mean and sample standard deviation for both the
                    # plotted curve and the reported final-epoch statistics.
                    display_epochs = aligned_run_perf.index.to_numpy()
                    display_metric_l, display_metric_m, display_metric_u = (
                        curve.to_numpy() for curve in get_metric_curves(aligned_run_perf)
                    )
                    exe_plot = ax.plot(display_epochs, display_metric_m,
                                       linestyle='-',
                                       linewidth=lwidth,
                                       c=color, alpha=1, clip_on=True, label=label,
                                       zorder=1)
                    if (row_idx,col_idx) == (0,0):
                        im = exe_plot
                    ax.fill_between(display_epochs,
                                    display_metric_l,
                                    display_metric_u,
                                    color=color, alpha=0.3, clip_on=True, edgecolor='none')

                    # These are the exact seed values underlying the final plotted point.
                    final_values = aligned_run_perf.iloc[-1].dropna()
                    counter = int(final_values.size)
                    metric_min = final_values.min()
                    metric_max = final_values.max()
                    metric_mid = (metric_min + metric_max) / 2
                    metric_median = final_values.median()
                    metric_mean = final_values.mean()
                    metric_std = final_values.std(ddof=1) if counter > 1 else 0.0
                    summary_row = [
                        metric, is_op, model_type, alpha, display_epochs[-1],
                        metric_min, metric_max, metric_mid, metric_median,
                        metric_mean, metric_std, counter,
                    ]
                    row_stats.append(summary_row)
                    summary_rows.append(summary_row)

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.set_xlim([start_epoch,end_epoch + 5])
                    # ax.set_xticks(list(range(25,126,25)))
                if not is_fns:
                    break  # only do once if model is not FNS type

        panel_summary_stats = pd.DataFrame(data=row_stats, columns=summary_colnames)

        # print message
        print(metric)
        print(f'is_op = {is_op}, qk_share = {qk_share}')
        print(panel_summary_stats.round(3))
        print('\n')                    

    summary_stats = pd.DataFrame(data=summary_rows, columns=summary_colnames)

    for cidx in range(ncols):
        # axs[0,cidx].set_ylim(62,80)
        axs[0,cidx].set_ylim(70,77.5)
        axs[0,cidx].margins(25)
        # axs[0,cidx].set_yticks(list(range(65, 81,5)))
        axs[1,cidx].set_ylim([0.7,1.1])
        axs[1,cidx].set_yticks(np.arange(0.7,1.1,0.1))
        axs[1,cidx].margins(25)

        display_epochs = list(range(30,110+1,20))
        axs[0,cidx].set_xticks(display_epochs)
        axs[0,cidx].set_xticklabels(display_epochs)
        axs[1,cidx].set_xticks(display_epochs)
        axs[1,cidx].set_xticklabels(display_epochs)

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
    axs[0,0].legend(loc='best', ncols=2, frameon=False)                     
    # for alpha in selected_alphas[::-1]:
    #     axs[0,0].plot([], [], c=HYP_CMAP(HYP_CNORM(alpha)), linestyle='solid', 
    #                   label=rf'$\alpha$ = {alpha}')         
    # ncol_legend = 2  #if len(model_types_plotted) == 3 else 1
    # if len(model_types_plotted) >= 2:
    #     #axs[0,0].legend(loc='best', ncol=ncol_legend, frameon=False)           
    #     axs[0,0].legend(loc='best', ncol=ncol_legend, frameon=False)                     

    # Add shared x and y labels     
    #fig.supxlabel('Epochs', fontsize='medium'); fig.supylabel(NAMES_DICT[metrics[0]], fontsize='medium')

    for row_idx in range(len(qk_shares)):        
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

    # Subfigure labels
    for ii, ax in enumerate(axs.flatten()):
        ax.text(-0.1, 1.15, rf"$\mathbf{{{ascii_lowercase[ii]}}}$",
            transform=ax.transAxes, ha='left',  va='top',
            usetex=False)

    # Adjust layout
    plt.subplots_adjust(wspace=0.4, hspace=0.3)            

    dataset_name_short = ''
    if isinstance(selected_dataset,str):
        if '_' in selected_dataset:
            for s in selected_dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += selected_dataset

    model_types_short = [model_type.replace(MODEL_SUFFIX,'') for model_type in model_types_plotted]
    dataset_name_short = ''
    if isinstance(selected_dataset,str):
        if '_' in selected_dataset:
            for s in selected_dataset.split('_'):
                dataset_name_short += s[0]
        else:
            dataset_name_short += selected_dataset

    model_types_short = [model_type.replace(MODEL_SUFFIX,'') for model_type in model_types_plotted]

    plt.tight_layout()

    from constants import FIGS_DIR
    SAVE_DIR = njoin(FIGS_DIR, 'vit-task')
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


def fna_alpha_effects(models_root, selected_dataset='cifar10',
                      fns_manifold='rd', qk_share=False, selected_alphas='none',
                      bandwidth=1, metric='val_acc', is_ops=[False, True],
                      display=False):
    global summary_stats, run_perf_all, plot_rows, seed_values

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

    end_epoch = 110

    # collect subdirs containing the model directories
    model_root_dirs = find_subdirs(njoin(models_root), MODEL_SUFFIX)
    print(model_root_dirs)

    # all trained model types
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

    def last_recorded_seed_values(run_perf_all):
        rows = []
        for seed, seed_run in run_perf_all.items():
            seed_run = seed_run.dropna()
            # print(f'seen_run shape: {seed_run.shape}')
            if len(seed_run) > 0:
                # rows.append({'seed': seed, metric: seed_run.iloc[-1]})
                rows.append({'seed': seed, metric: seed_run.iloc[end_epoch-1]})
        return pd.DataFrame(rows)

    ncols = len(is_ops)
    fig, axs = plt.subplots(1, ncols, figsize=(2.5*ncols, 2.5), 
                            sharex=True, sharey=True, squeeze=False)
    axs = axs[0]
    row_stats = []
    marker_color = '#2E63A6'
    marker_shape = 'o'

    for col_idx, is_op in enumerate(is_ops):
        ax = axs[col_idx]
        model_type = ('op' if is_op else '') + fns_model_type
        if model_type not in DCT_ALL:
            continue

        df_model = DCT_ALL[model_type]
        condition = (df_model['ensembles'] > 0) & (df_model['qk_share'] == qk_share) &\
                    (df_model['is_op'] == is_op) &\
                    ((df_model['dataset_name'] == selected_dataset) |
                     (df_model['model_dir'].str.contains(selected_dataset))) &\
                    (df_model['model_dir'].str.contains(f'{model_type}-'))
        matching_df = df_model[condition].copy()
        if bandwidth is not None:
            matching_df = matching_df[matching_df['bandwidth'] == bandwidth]

        plot_rows = []
        for alpha in selected_alphas:
            model_info = matching_df[matching_df['alpha'] == alpha]
            seed_values_all = []
            for _, model_row in model_info.iterrows():
                training_seeds = normalize_training_seeds(model_row['seeds'])
                if len(training_seeds) == 0:
                    continue
                _, run_perf_all = load_seed_runs(model_row['model_dir'], training_seeds, metric)
                if run_perf_all is None:
                    continue
                seed_values = last_recorded_seed_values(run_perf_all)
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

            row_stats.append([model_type, alpha, is_op, qk_share, bandwidth,
                              mean, std, sem, median, min_val, max_val, counter])
            plot_rows.append([alpha, mean, std, counter])

        if len(plot_rows) == 0:
            continue

        plot_df = pd.DataFrame(plot_rows, columns=['alpha', 'mean', 'std', 'counter']).sort_values('alpha')
        x = plot_df['alpha'].to_numpy()
        y = plot_df['mean'].to_numpy()
        yerr = plot_df['std'].fillna(0).to_numpy()

        ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor=marker_color,
                    elinewidth=0.8, capsize=2, zorder=1)
        ax.scatter(x, y, marker=marker_shape, c=marker_color,
                   s=36, edgecolor='white', linewidth=0.5, zorder=2)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(r'$\alpha$')
        ax.set_xticks(selected_alphas)
        ax_title = r'$\mathbf{W}_{Q,K} \in O(d)$' if is_op else r'$\mathbf{W}_{Q,K} \notin O(d)$'
        ax.set_title(ax_title)

    # Subfigure labels
    for ii, ax in enumerate(axs.flatten()):
        ax.text(-0.1, 1.15, rf"$\mathbf{{{ascii_lowercase[ii]}}}$",
            transform=ax.transAxes, ha='left',  va='top',
            usetex=False)

    axs[0].set_ylabel('Testing accuracy (%)' if 'acc' in metric else NAMES_DICT.get(metric, metric))
    summary_stats = pd.DataFrame(
        data=row_stats,
        columns=['model_type', 'alpha', 'is_op', 'qk_share', 'bandwidth',
                 'mean', 'std', 'sem', 'median', 'min', 'max', 'counter']
    )
    print(metric)
    print(f'qk_share = {qk_share}, bandwidth = {bandwidth}')
    print(summary_stats.round(3))
    print('\n')

    plt.tight_layout()

    from constants import FIGS_DIR
    SAVE_DIR = njoin(FIGS_DIR, 'vit-task')
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
